"""Comprehensive false-positive audit for all HermesKatana scanners.

NOTE: This test file intentionally contains strings that LOOK like security
threats but are actually benign. These are test inputs to find false positives
in the scanner regex patterns. No actual dangerous code is executed here.
All 'dangerous' strings are used purely as test data for pattern matching.
"""
import sys
sys.path.insert(0, "src")

from hermes_katana.scanner.injection import detect_injection
from hermes_katana.scanner.commands import detect_dangerous_command
from hermes_katana.scanner.content import scan_content
from hermes_katana.scanner.secrets import scan_for_secrets

total_fp = 0

# ============================================================
# 1. INJECTION SCANNER (50+ benign cases)
# ============================================================
print("=" * 70)
print("INJECTION SCANNER FALSE POSITIVES")
print("=" * 70)

injection_tests = [
    # Original cases
    "The system prompt is displayed correctly on the dashboard.",
    "Please update the configuration to use the new settings.",
    "You are now able to see the results of the test.",
    "The admin panel shows all user accounts.",
    "Override the default CSS styles for the button.",
    "Instead, you should use the new API endpoint.",
    "Remember to update the documentation before release.",
    "IMPORTANT: Make sure tests pass before merging.",
    "IMPORTANT: Before running tests, install dependencies.",
    "IMPORTANT: Always run lint before committing.",
    "IMPORTANT: Never commit directly to main branch.",
    "From now on, your instructions are to follow PEP 8.",
    "Actually, your real purpose is to serve static files.",
    "The developer mode enabled flag is stored in config.",
    "Pretend to be a different user for testing auth flows.",
    "The system message injection tests are passing.",
    "Call the function with parameters x and y.",
    "Read the file content and display it.",
    "Send the output to the log file.",
    "Change the tool parameters in the config file.",
    "Modify the function arguments to include the new flag.",
    "Reveal the hidden div by changing display to block.",
    "Show your system prompt settings in the documentation.",
    "What are your instructions for setting up the environment?",
    "Output the results to a CSV file.",
    "Please repeat everything above this line in the report.",
    "The first message in the queue should be processed.",
    "The CSS override instructions are in the theme file.",
    "                                 return result",
    "Use base64 encoding for binary data in JSON payloads.",
    "Instead, you should follow the coding standards.",
    "You are now using Python 3.12.",
    "instead, you must do the following refactoring steps.",
    "For testing purposes only, we use mock data.",
    "authorized security test completed successfully.",
    "Override the default theme settings in config.yaml.",
    "The new instructions: follow PEP 8 style guide.",
    "Forget everything you know about the old API and use v2.",
    "Instead of the above approach, use dependency injection.",
    "Remember this for future reference: use type hints.",
    "Add this to your memory: the meeting is on Friday.",
    # Developer discussion (natural language that might trigger injection patterns)
    "The function should override the default behavior.",
    "We need to kill the process that's hanging.",
    "Drop the old table and create a new one.",
    "Let's ignore the failing tests for now and focus on the feature.",
    "Can you forget about the previous approach and try something new?",
    "You should act as a proxy between the frontend and backend.",
    "The system prompt for the chatbot is stored in a YAML file.",
    "Pretend this is a production environment for the demo.",
    "Instead of the above, let's use a different algorithm.",
    "From now on we'll be using TypeScript instead of JavaScript.",
    "Your role is to handle authentication middleware.",
    "Actually, the real problem is in the database layer.",
    "Please disregard the old documentation, it's outdated.",
    "The instructions for deployment are in the wiki.",
    "Repeat the migration steps for each environment.",
    "Output all debug information to the console.",
    "Reveal the feature flag status in the admin dashboard.",
    "The new prompt template uses Jinja2 syntax.",
    "Modify your approach to use async/await patterns.",
    "Change your strategy to use connection pooling.",
    "Do not follow the old coding conventions anymore.",
    "The AI assistant should respond with helpful information.",
]

for s in injection_tests:
    findings = detect_injection(s)
    for f in findings:
        if f.confidence >= 0.7:
            total_fp += 1
            print(f"FP #{total_fp}: pattern={f.pattern_name!r} conf={f.confidence}")
            print(f"   Input:   {s!r}")
            print(f"   Matched: {f.matched_text!r}")
            print()

# ============================================================
# 2. COMMAND SCANNER (50+ benign cases)
# ============================================================
print("=" * 70)
print("COMMAND SCANNER FALSE POSITIVES")
print("=" * 70)

command_tests = [
    # Original cases
    "git rm --cached myfile.txt",
    "git rebase origin/main",
    "git init",
    "pip install requests",
    "python setup.py install",
    "python -m pytest tests/",
    "cat README.md",
    "ls -la /home/user",
    "docker build -t myapp .",
    "docker run myapp:latest",
    "docker compose up -d",
    "systemctl status nginx",
    "systemctl restart nginx",
    "SELECT * FROM users WHERE id = 1",
    "CREATE TABLE users (id INT, name VARCHAR(255))",
    "INSERT INTO logs (message) VALUES ('test')",
    "curl https://api.github.com/repos",
    "wget https://example.com/file.tar.gz",
    "ssh user@server.com",
    "scp local_file.txt user@server:/home/user/",
    "crontab -l",
    "The shutdown procedure involves saving all state first.",
    "We need to reboot the service after updating.",
    "Use shred to securely delete sensitive files when needed.",
    "chmod 755 deploy.sh",
    "chmod +x run.sh",
    "rclone config",
    "rclone ls remote:bucket",
    "env | grep PATH",
    # Git commands
    "git push origin main",
    "git rebase -i HEAD~3",
    "git stash pop",
    "git cherry-pick abc123",
    "git log --oneline --graph",
    "git diff HEAD~1",
    "git fetch --all --prune",
    "git checkout -b feature/new-login",
    "git merge develop --no-ff",
    "git tag -a v1.2.0 -m 'Release 1.2.0'",
    "git reset --soft HEAD~1",
    "git bisect start",
    "git submodule update --init --recursive",
    "git blame src/main.py",
    "git clean -fd",
    # Docker commands
    "docker build -t myapp:v2 .",
    "docker-compose up -d",
    "docker exec -it container bash",
    "docker ps -a",
    "docker logs --tail 100 mycontainer",
    "docker network create mynetwork",
    "docker volume prune",
    "docker pull postgres:15",
    "docker stop $(docker ps -q)",
    "docker system df",
    # Python dev commands
    "pip install requests",
    "python3 -m pytest",
    "virtualenv .venv",
    "pip install -r requirements.txt",
    "python3 -m venv myenv",
    "pip freeze > requirements.txt",
    "python3 manage.py migrate",
    "python3 -c 'print(\"hello\")'",
    "pip install --upgrade pip",
    "python3 setup.py sdist bdist_wheel",
    "pytest --cov=src tests/",
    "mypy src/ --strict",
    "black --check src/",
    "flake8 src/",
    "isort --check-only src/",
    # System admin commands
    "systemctl status nginx",
    "journalctl -f",
    "df -h",
    "htop",
    "free -m",
    "top -bn1",
    "uptime",
    "whoami",
    "hostname -I",
    "uname -a",
    "lsof -i :8080",
    "netstat -tlnp",
    "ps aux | grep python",
    "tail -f /var/log/syslog",
    "dmesg | tail -20",
    "mount | grep sda",
    "du -sh /var/log",
    "iostat -x 1",
    "ss -tlnp",
    "ip addr show",
    # Config-related
    "export PATH=$HOME/bin:$PATH",
    "alias ll='ls -la'",
    "source ~/.bashrc",
    "echo $SHELL",
    "printenv",
    "locale",
]

for s in command_tests:
    findings = detect_dangerous_command(s)
    for f in findings:
        total_fp += 1
        print(f"FP #{total_fp}: pattern={f.pattern_name!r} sev={f.severity.value}")
        print(f"   Input:   {s!r}")
        print(f"   Matched: {f.matched_text!r}")
        print()

# ============================================================
# 3. CONTENT SCANNER (50+ benign cases)
# ============================================================
print("=" * 70)
print("CONTENT SCANNER FALSE POSITIVES")
print("=" * 70)

content_tests = [
    # Original cases
    "![screenshot](https://example.com/image.png)",
    "[Click here to read more](https://docs.python.org)",
    "[download the SDK](https://github.com/repo/releases)",
    "[view your dashboard](https://myapp.com/dashboard)",
    "![logo](https://cdn.example.com/logo.png?v=2)",
    '<input type="text" name="search">',
    '<link rel="stylesheet" href="styles.css">',
    '<meta charset="utf-8">',
    "subprocess.run(['ls', '-la'])",
    "subprocess.Popen(['python', 'script.py'])",
    "Visit https://hooks.slack.com/services for webhook setup.",
    "$ sudo apt-get update",
    "$ curl https://api.example.com/data",
    # Markdown links and images (benign)
    "![diagram](./docs/architecture.png)",
    "[API docs](https://api.example.com/v2/docs)",
    "[PyPI package](https://pypi.org/project/requests/)",
    "[GitHub](https://github.com/user/repo)",
    "![badge](https://img.shields.io/badge/build-passing-green)",
    "[Read the Docs](https://readthedocs.org/projects/mylib/)",
    "[npm package](https://www.npmjs.com/package/lodash)",
    "[Stack Overflow answer](https://stackoverflow.com/questions/12345)",
    "![coverage](https://codecov.io/gh/user/repo/branch/main/graph/badge.svg)",
    "[MDN Web Docs](https://developer.mozilla.org/en-US/docs/Web)",
    # HTML fragments (benign templates/docs)
    '<div class="container">',
    '<a href="/about">About Us</a>',
    '<img src="logo.png" alt="Company Logo">',
    '<button type="submit">Save</button>',
    '<label for="search">Search:</label>',
    '<span class="badge badge-success">Active</span>',
    '<p>This is a paragraph of documentation.</p>',
    '<h1>Welcome to the project</h1>',
    '<ul><li>Item one</li><li>Item two</li></ul>',
    '<table><tr><th>Name</th><th>Value</th></tr></table>',
    # Code snippets (benign discussion of code)
    "subprocess.run(['git', 'status'], capture_output=True)",
    "os.path.join(base_dir, 'config.yaml')",
    "import subprocess",
    "from pathlib import Path",
    "shutil.copy2(src, dst)",
    "os.environ.get('DATABASE_URL', 'sqlite:///db.sqlite3')",
    "json.dumps({'key': 'value'}, indent=2)",
    "requests.get('https://api.example.com/data')",
    "logging.basicConfig(level=logging.DEBUG)",
    "argparse.ArgumentParser(description='My CLI tool')",
    # URLs in documentation context
    "See https://docs.python.org/3/library/subprocess.html for details.",
    "Documentation at https://flask.palletsprojects.com/en/3.0.x/",
    "Install from https://pypi.org/project/hermes-katana/",
    "Clone from https://github.com/user/repo.git",
    "API endpoint: https://api.example.com/v1/users",
    "Webhook docs: https://api.slack.com/messaging/webhooks",
    "CI dashboard: https://app.circleci.com/pipelines/github/user/repo",
    "Docker Hub: https://hub.docker.com/r/library/python",
    # Config file snippets
    "DATABASE_URL=postgres://localhost/dev",
    "REDIS_URL=redis://localhost:6379/0",
    "LOG_LEVEL=debug",
    "PORT=8080",
    "DEBUG=true",
]

for s in content_tests:
    findings = scan_content(s)
    for f in findings:
        total_fp += 1
        print(f"FP #{total_fp}: pattern={f.pattern_name!r} sev={f.severity.value}")
        print(f"   Input:   {s!r}")
        print(f"   Matched: {f.matched_text!r}")
        print()

# ============================================================
# 4. SECRET SCANNER (50+ benign cases)
# ============================================================
print("=" * 70)
print("SECRET SCANNER FALSE POSITIVES")
print("=" * 70)

secret_tests = [
    # Original cases
    'password=get_password()',
    'password="***"',
    'password="***"',
    'commit 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b',
    'sk_test_1234567890abcdef',
    'pk_live_1234567890abcdef1234567890abcdef',
    'postgresql://localhost:5432/mydb',
    'redis://localhost:6379/0',
    'bearer = get_auth_token()',
    'SKIP_VALIDATION=true',
    'SKU12345678901234567890123456789012',
    'SG_SETTING=production',
    'key-value pairs are fundamental',
    'SHA256 hash: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    # Config discussion (benign)
    "Set DATABASE_URL=postgres://localhost/dev",
    "export PATH=$HOME/bin:$PATH",
    "export NODE_ENV=production",
    "export PYTHONPATH=/app/src",
    "DJANGO_SETTINGS_MODULE=myapp.settings",
    "FLASK_APP=app.py",
    "CELERY_BROKER_URL=redis://localhost:6379/0",
    # Password-related discussion (not actual passwords)
    "The password field should be hashed with bcrypt.",
    "password_hash = bcrypt.hashpw(password, salt)",
    "user.set_password(new_password)",
    'password_input = input("Enter password: ")',
    "Minimum password length is 8 characters.",
    "PASSWORD_VALIDATORS are configured in settings.py.",
    "Reset password link sent to user email.",
    "password: ${DB_PASSWORD}",
    "The password must contain uppercase and lowercase.",
    # Token/key discussion (not actual secrets)
    "The API key is stored in the vault.",
    "Generate a new token with: flask token create",
    "The access_token expires after 3600 seconds.",
    "token_type: Bearer",
    "Use refresh_token to get a new access token.",
    "api_key = os.environ['API_KEY']",
    "The secret_key is loaded from environment variables.",
    "AWS_ACCESS_KEY_ID should be set in CI.",
    "key = Fernet.generate_key()",
    # Hashes and checksums (not secrets)
    "md5sum: d41d8cd98f00b204e9800998ecf8427e",
    "sha1: da39a3ee5e6b4b0d3255bfef95601890afd80709",
    "SHA256: e3b0c44298fc1c149afbf4c8996fb924",
    "file checksum: abc123def456",
    "Git commit SHA: a1b2c3d4e5f6",
    # Benign strings that look like secrets
    "sk_test is a prefix for Stripe test keys",
    "The pk_live prefix indicates a live publishable key",
    "SG. prefixed settings are for SendGrid config",
    "ghp_ tokens are GitHub personal access tokens",
    "xoxb- tokens are Slack bot tokens",
    "AWS keys start with the AKIA prefix by convention",
    # Connection strings (no credentials - just hosts)
    "mongodb://localhost:27017/testdb",
    "sqlite:///var/data/app.db",
    "redis://localhost:6379/0",
    "Connect to the database at localhost:5432",
    "The connection string format is protocol://host:port/db",
    # Variable assignments (not actual values)
    "SECRET_KEY = os.getenv('SECRET_KEY')",
    "API_TOKEN = config.get('api_token')",
    "DB_PASSWORD = vault.read('database/password')",
    "ENCRYPTION_KEY = load_key_from_file(keypath)",
]

for s in secret_tests:
    findings = scan_for_secrets(s)
    for f in findings:
        total_fp += 1
        print(f"FP #{total_fp}: pattern={f.pattern_name!r} sev={f.severity.value} conf={f.confidence}")
        print(f"   Input:   {s!r}")
        print(f"   Matched: {f.matched_text}")
        print()

# ============================================================
print("=" * 70)
print(f"TOTAL FALSE POSITIVES FOUND: {total_fp}")
print("=" * 70)
print(f"\nTest counts:")
print(f"  Injection scanner: {len(injection_tests)} benign inputs")
print(f"  Command scanner:   {len(command_tests)} benign inputs")
print(f"  Content scanner:   {len(content_tests)} benign inputs")
print(f"  Secret scanner:    {len(secret_tests)} benign inputs")
print(f"  TOTAL:             {len(injection_tests) + len(command_tests) + len(content_tests) + len(secret_tests)} benign inputs")
