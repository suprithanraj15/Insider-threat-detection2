
[CmdletBinding()]
param(
    [switch]$Once,
    [int]$BatchSize = 200,
    [int]$PollSeconds = 5
)

$ErrorActionPreference = "Stop"

$ApiUrl = "http://127.0.0.1:8000/api/events"
$LogName = "System"

$ProjectRoot = if ($PSScriptRoot) {
    (Resolve-Path "$PSScriptRoot\..\..").Path
} else {
    (Get-Location).Path
}
$CheckpointDir = Join-Path $ProjectRoot "data\outputs"
$CheckpointFile = Join-Path $CheckpointDir "system_event_checkpoint.txt"

Write-Host "Windows System Event Collector"
Write-Host "Source: $LogName"
Write-Host "API: $ApiUrl"
Write-Host "Poll interval: $PollSeconds seconds"
Write-Host "Stop with Ctrl+C"
Write-Host ""

if (-not (Test-Path $CheckpointDir)) {
    New-Item -ItemType Directory -Path $CheckpointDir -Force | Out-Null
}

# Confirm the API is reachable before starting.
try {
    Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" `
        -Method Get -ErrorAction Stop | Out-Null
}
catch {
    Write-Error "FastAPI is unavailable. Start the API before running the collector."
    exit 1
}

# Initialize the checkpoint from previously ingested System events.
# This avoids importing the entire historical Windows System log.
if (-not (Test-Path $CheckpointFile)) {
    try {
        $existing = Invoke-RestMethod `
            -Uri "$ApiUrl`?limit=500" `
            -Method Get -ErrorAction Stop

        $recordIds = @(
            foreach ($item in $existing.events) {
                if ($item.source_event_id -match '^System-(\d+)$') {
                    [long]$Matches[1]
                }
            }
        )

        if ($recordIds.Count -gt 0) {
            $lastRecordId = ($recordIds | Measure-Object -Maximum).Maximum
        }
        else {
            $latest = Get-WinEvent -LogName $LogName -MaxEvents 1 `
                -ErrorAction Stop
            $lastRecordId = [long]$latest.RecordId
        }

        Set-Content -Path $CheckpointFile -Value $lastRecordId
        Write-Host "Initialized checkpoint at record ID $lastRecordId"
        Write-Host "Collecting subsequent events only."
    }
    catch {
        Write-Error "Could not initialize the checkpoint: $($_.Exception.Message)"
        exit 1
    }
}
else {
    try {
        $lastRecordId = [long](Get-Content $CheckpointFile -Raw).Trim()
    }
    catch {
        Write-Error "Checkpoint is invalid. No events were processed."
        exit 1
    }

    Write-Host "Resuming after record ID $lastRecordId"
}

while ($true) {
    try {
        $xpath = "*[System[EventRecordID > $lastRecordId]]"

        $events = @(
            Get-WinEvent -LogName $LogName -FilterXPath $xpath `
                -MaxEvents $BatchSize -ErrorAction SilentlyContinue |
                Sort-Object RecordId
        )

        foreach ($event in $events) {
            $payload = @{
                source_event_id = "$LogName-$($event.RecordId)"
                event_timestamp = $event.TimeCreated.ToUniversalTime().ToString("o")
                host_name       = $env:COMPUTERNAME
                user_name       = $env:USERNAME
                event_source    = $event.ProviderName
                event_type      = if ($event.LevelDisplayName) {
                    $event.LevelDisplayName
                } else {
                    "Unknown"
                }
                event_code      = [string]$event.Id
                details         = @{
                    log_name  = $LogName
                    record_id = $event.RecordId
                    level     = $event.LevelDisplayName
                    message   = $event.Message
                }
            } | ConvertTo-Json -Depth 5

            try {
                $response = Invoke-RestMethod `
                    -Uri $ApiUrl `
                    -Method Post `
                    -ContentType "application/json" `
                    -Body $payload `
                    -ErrorAction Stop

                if ($response.duplicate -eq $true) {
                    Write-Host "Duplicate skipped: $($event.RecordId)"
                }
                else {
                    Write-Host "Stored event $($event.RecordId): database ID $($response.event_id)"
                }

                # Advance only after successful storage or confirmed duplicate.
                $lastRecordId = [long]$event.RecordId
                Set-Content -Path $CheckpointFile -Value $lastRecordId
            }
            catch {
                Write-Warning "Event $($event.RecordId) failed; will retry. $($_.Exception.Message)"

                if ($_.ErrorDetails.Message) {
                    Write-Warning "API response: $($_.ErrorDetails.Message)"
                }

                # Do not skip a failed event.
                break
            }
        }
    }
    catch {
        Write-Warning "Windows log read failed: $($_.Exception.Message)"
    }

    if ($Once) {
        Write-Host "Collection pass completed (-Once specified). Exiting."
        break
    }

    Start-Sleep -Seconds $PollSeconds
}
