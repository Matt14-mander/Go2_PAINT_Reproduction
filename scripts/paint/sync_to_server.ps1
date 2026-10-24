<#
Sync this working tree (including uncommitted changes and materialized LFS assets).
Requires local tar + OpenSSH; uses SSH stdin, so remote SFTP/scp is not required.
This script uploads only. It never installs dependencies or starts training.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$SshTarget,
    [string]$RemoteDirectory = '/workspace/Go2_PAINT_Reproduction',
    [ValidateRange(1, 65535)][int]$Port = 22,
    [string]$IdentityFile
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
# Restrict shell-interpreted values to simple, absolute Linux paths.
if ($RemoteDirectory -notmatch '^/[A-Za-z0-9_./-]+$' -or
    @($RemoteDirectory.Split('/') | Where-Object { $_ -and $_ -ne '.' }).Count -eq 0 -or
    $RemoteDirectory.Split('/') -contains '..') {
    throw 'RemoteDirectory must be an absolute Linux path without spaces, shell characters or .. segments.'
}
if ($SshTarget.StartsWith('-') -or $SshTarget -match '[\s\x00]') {
    throw 'SshTarget must be an SSH config alias or user@host.'
}
$sshPath = (Get-Command ssh -CommandType Application).Source
$tarPath = (Get-Command tar -CommandType Application).Source
$archive = Join-Path ([System.IO.Path]::GetTempPath()) ('paint-sync-' + [guid]::NewGuid() + '.tar.gz')
try {
    $tarArgs = @('-czf', $archive,
        '--exclude=.git', '--exclude=logs', '--exclude=runs', '--exclude=outputs',
        '--exclude=__pycache__', '--exclude=*.pyc', '--exclude=.venv', '--exclude=venv',
        '--exclude=.codex', '--exclude=.agents', '--exclude=.aws', '--exclude=.ssh',
        '--exclude=.env', '--exclude=.env.*', '--exclude=wandb', '--exclude=recordings',
        '--exclude=node_modules', '-C', $repo, '.')
    & $tarPath @tarArgs
    if ($LASTEXITCODE -ne 0) { throw 'tar archive creation failed.' }
    Write-Host ('Uploading {0:N1} MB to {1}:{2}' -f ((Get-Item $archive).Length / 1MB), $SshTarget, $RemoteDirectory)
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $sshPath
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardInput = $true
    foreach ($arg in @('-T', '-o', 'BatchMode=yes', '-p', "$Port")) { $startInfo.ArgumentList.Add($arg) }
    if ($IdentityFile) {
        $startInfo.ArgumentList.Add('-i')
        $startInfo.ArgumentList.Add((Resolve-Path -LiteralPath $IdentityFile).Path)
    }
    $startInfo.ArgumentList.Add($SshTarget)
    $startInfo.ArgumentList.Add("mkdir -p -- '$RemoteDirectory' && tar -xzf - -C '$RemoteDirectory'")
    $process = [System.Diagnostics.Process]::Start($startInfo)
    $stream = [System.IO.File]::OpenRead($archive)
    try { $stream.CopyTo($process.StandardInput.BaseStream) }
    finally { $stream.Dispose(); $process.StandardInput.Close() }
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) { throw "SSH extraction failed (exit $($process.ExitCode))." }
    Write-Host "Synced to $RemoteDirectory. Existing server logs and checkpoints were not removed."
    Write-Host 'Next: activate the server Isaac Lab environment and run scripts/paint/server_train.py.'
}
finally {
    if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive }
}
