#!/usr/bin/env node
/**
 * GPU Worker CLI - Node.js 入口
 * 包装 Python Worker，提供简单的 npm/npx 安装体验
 */

const { Command } = require('commander');
const chalk = require('chalk');
const ora = require('ora');
const inquirer = require('inquirer');
const { spawn, execSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const which = require('which');

const PACKAGE_DIR = path.resolve(__dirname, '..');
const PYTHON_DIR = PACKAGE_DIR;
const CONFIG_FILE = path.join(process.cwd(), 'config.yaml');

// 检测 Python
function findPython() {
    const pythonCommands = ['python3', 'python', 'py'];

    for (const cmd of pythonCommands) {
        try {
            const pythonPath = which.sync(cmd);
            // 验证版本
            const version = execSync(`${cmd} --version`, { encoding: 'utf8' });
            const match = version.match(/Python (\d+)\.(\d+)/);
            if (match && parseInt(match[1]) >= 3 && parseInt(match[2]) >= 9) {
                return cmd;
            }
        } catch (e) {
            continue;
        }
    }
    return null;
}

// 检查虚拟环境
function getVenvPython() {
    const venvPath = path.join(PACKAGE_DIR, '.venv');

    if (process.platform === 'win32') {
        const pythonPath = path.join(venvPath, 'Scripts', 'python.exe');
        if (fs.existsSync(pythonPath)) return pythonPath;
    } else {
        const pythonPath = path.join(venvPath, 'bin', 'python');
        if (fs.existsSync(pythonPath)) return pythonPath;
    }
    return null;
}

function detectCudaVersion() {
    try {
        const output = execSync('nvidia-smi', { encoding: 'utf8' });
        const match = output.match(/CUDA Version:\\s*(\\d+)\\.(\\d+)/);
        if (!match) return null;
        return { major: parseInt(match[1], 10), minor: parseInt(match[2], 10) };
    } catch (e) {
        return null;
    }
}

function selectTorchIndexUrl(cudaVersion) {
    if (!cudaVersion) return null;
    const versionValue = cudaVersion.major * 100 + cudaVersion.minor;
    if (versionValue >= 1204) return 'https://download.pytorch.org/whl/cu124';
    if (versionValue >= 1201) return 'https://download.pytorch.org/whl/cu121';
    if (versionValue >= 1108) return 'https://download.pytorch.org/whl/cu118';
    return null;
}

function createRequirementsWithoutTorch(requirementsFile) {
    const content = fs.readFileSync(requirementsFile, 'utf8');
    const filtered = content
        .split(/\r?\n/)
        .filter((line) => {
            const trimmed = line.trim();
            if (!trimmed || trimmed.startsWith('#')) return true;
            return !/^torch([<>=!~].*)?$/.test(trimmed);
        });
    const filteredPath = path.join(PACKAGE_DIR, '.requirements.no-torch.txt');
    fs.writeFileSync(filteredPath, filtered.join('\n'));
    return filteredPath;
}

// 创建虚拟环境
async function createVenv(pythonCmd) {
    const spinner = ora('Creating Python virtual environment...').start();
    const venvPath = path.join(PACKAGE_DIR, '.venv');

    try {
        execSync(`${pythonCmd} -m venv "${venvPath}"`, { stdio: 'pipe' });
        spinner.succeed('Virtual environment created');
        return true;
    } catch (e) {
        spinner.fail('Failed to create virtual environment');
        console.error(chalk.red(e.message));
        return false;
    }
}

// 安装 Python 依赖
async function installDependencies() {
    const venvPython = getVenvPython();
    if (!venvPython) {
        console.error(chalk.red('Virtual environment not found'));
        return false;
    }

    const spinner = ora('Installing Python dependencies...').start();
    const requirementsFile = path.join(PACKAGE_DIR, 'requirements.txt');
    const cudaVersion = detectCudaVersion();
    const torchIndexUrl = selectTorchIndexUrl(cudaVersion);

    try {
        if (cudaVersion && !torchIndexUrl) {
            console.log(chalk.yellow(`检测到 CUDA ${cudaVersion.major}.${cudaVersion.minor}，无匹配的 PyTorch 版本，将安装 CPU 版。`));
        }
        if (torchIndexUrl) {
            execSync(`"${venvPython}" -m pip install torch --index-url "${torchIndexUrl}" --upgrade --force-reinstall -q`, {
                stdio: 'pipe',
                timeout: 600000  // 10分钟超时
            });
        } else {
            execSync(`"${venvPython}" -m pip install "torch>=2.0.0" -q`, {
                stdio: 'pipe',
                timeout: 600000  // 10分钟超时
            });
        }

        const filteredRequirements = createRequirementsWithoutTorch(requirementsFile);
        try {
            execSync(`"${venvPython}" -m pip install -r "${filteredRequirements}" -q`, {
                stdio: 'pipe',
                timeout: 600000  // 10分钟超时
            });
        } finally {
            if (fs.existsSync(filteredRequirements)) {
                fs.unlinkSync(filteredRequirements);
            }
        }

        spinner.succeed('Dependencies installed');
        return true;
    } catch (e) {
        spinner.fail('Failed to install dependencies');
        console.error(chalk.red(e.message));
        return false;
    }
}

// 运行 Python CLI
function runPythonCLI(args, pythonCmd) {
    const resolvedCmd = pythonCmd || getVenvPython() || findPython();

    if (!resolvedCmd) {
        console.error(chalk.red('Python 3.9+ not found!'));
        console.log(chalk.yellow('Please install Python 3.9 or higher:'));
        console.log('  - Windows: https://www.python.org/downloads/');
        console.log('  - macOS: brew install python@3.11');
        console.log('  - Linux: sudo apt install python3.11');
        process.exit(1);
    }

    const cliPath = path.join(PACKAGE_DIR, 'cli.py');

    const proc = spawn(resolvedCmd, [cliPath, ...args], {
        stdio: 'inherit',
        cwd: process.cwd()
    });

    proc.on('close', (code) => {
        process.exit(code);
    });

    proc.on('error', (err) => {
        console.error(chalk.red('Failed to start Python process:'), err.message);
        process.exit(1);
    });
}

// 初始化检查

// 主程序
async function ensureSetup(options = {}) {
    const allowPrompt = Boolean(options.allowPrompt);
    let useSystemPython = Boolean(options.useSystemPython);
    let skipInstall = Boolean(options.skipInstall);

    if (useSystemPython) {
        const pythonCmd = findPython();
        if (!pythonCmd) {
            console.error(chalk.red('Python 3.9+ is required but not found!'));
            console.log(chalk.yellow('\nPlease install Python:'));
            console.log('  - Windows: https://www.python.org/downloads/');
            console.log('  - macOS: brew install python@3.11');
            console.log('  - Linux: sudo apt install python3.11');
            process.exit(1);
        }

        console.log(chalk.green(`Using system Python: ${pythonCmd}`));
        return { pythonCmd, useSystemPython: true };
    }

    let venvPython = getVenvPython();
    let createdVenv = false;

    if (!venvPython) {
        if (allowPrompt) {
            const { mode } = await inquirer.prompt([{
                type: 'list',
                name: 'mode',
                message: '未检测到虚拟环境，请选择运行方式:',
                choices: [
                    { name: '使用系统 Python（跳过虚拟环境与依赖安装）', value: 'system' },
                    { name: '创建虚拟环境并安装依赖（推荐）', value: 'venv' },
                    { name: '创建虚拟环境但跳过依赖安装', value: 'venv-skip-install' }
                ]
            }]);

            if (mode === 'system') {
                useSystemPython = true;
                const pythonCmd = findPython();
                if (!pythonCmd) {
                    console.error(chalk.red('Python 3.9+ is required but not found!'));
                    console.log(chalk.yellow('\nPlease install Python:'));
                    console.log('  - Windows: https://www.python.org/downloads/');
                    console.log('  - macOS: brew install python@3.11');
                    console.log('  - Linux: sudo apt install python3.11');
                    process.exit(1);
                }

                console.log(chalk.green(`Using system Python: ${pythonCmd}`));
                return { pythonCmd, useSystemPython: true };
            }

            if (mode === 'venv-skip-install') {
                skipInstall = true;
            }
        }

        console.log(chalk.cyan('First time setup detected. Setting up environment...\n'));

        const pythonCmd = findPython();
        if (!pythonCmd) {
            console.error(chalk.red('Python 3.9+ is required but not found!'));
            console.log(chalk.yellow('\nPlease install Python:'));
            console.log('  - Windows: https://www.python.org/downloads/');
            console.log('  - macOS: brew install python@3.11');
            console.log('  - Linux: sudo apt install python3.11');
            process.exit(1);
        }

        console.log(chalk.green(`Found Python: ${pythonCmd}`));

        if (!await createVenv(pythonCmd)) {
            process.exit(1);
        }

        createdVenv = true;
        venvPython = getVenvPython();
    }

    if (createdVenv && !skipInstall) {
        if (!await installDependencies()) {
            process.exit(1);
        }

        console.log(chalk.green('\nSetup complete!\n'));
    }

    return { pythonCmd: venvPython || findPython(), useSystemPython: false };
}

function addCommonOptions(command) {
    return command
        .option('--use-system-python', '使用系统 Python（跳过虚拟环境与依赖安装）')
        .option('--skip-install', '跳过依赖安装，仅使用已有环境');
}

const program = new Command();

program
    .name('gpu-worker')
    .description('分布式GPU推理 Worker 节点')
    .version('1.0.0')
    .option('--use-system-python', '使用系统 Python（跳过虚拟环境与依赖安装）')
    .option('--skip-install', '跳过依赖安装，仅使用已有环境');

addCommonOptions(
    program
        .command('install')
)
    .description('安装/更新 Python 依赖')
    .action(async function () {
        const opts = { ...program.opts(), ...this.opts() };
        if (opts.useSystemPython) {
            console.log(chalk.yellow('install 仅针对虚拟环境，系统 Python 请自行安装依赖。'));
            return;
        }
        await ensureSetup({ ...opts, useSystemPython: false, skipInstall: true, allowPrompt: false });
        await installDependencies();
    });

addCommonOptions(
    program
        .command('configure')
)
    .description('交互式配置向导')
    .action(async function () {
        const opts = { ...program.opts(), ...this.opts() };
        const setup = await ensureSetup({ ...opts, allowPrompt: false });
        runPythonCLI(['configure'], setup.pythonCmd);
    });

addCommonOptions(
    program
        .command('start')
)
    .description('启动 Worker')
    .option('-c, --config <path>', '配置文件路径', 'config.yaml')
    .action(async function () {
        const cmdOpts = this.opts();
        const opts = { ...program.opts(), ...cmdOpts };
        const setup = await ensureSetup({ ...opts, allowPrompt: false });

        // 检查配置文件
        const configPath = path.resolve(cmdOpts.config);
        if (!fs.existsSync(configPath)) {
            console.log(chalk.yellow('No config file found. Starting configuration wizard...\n'));
            runPythonCLI(['configure'], setup.pythonCmd);
            return;
        }

        runPythonCLI(['start', '-c', configPath], setup.pythonCmd);
    });

addCommonOptions(
    program
        .command('status')
)
    .description('查看状态')
    .action(async function () {
        const opts = { ...program.opts(), ...this.opts() };
        const setup = await ensureSetup({ ...opts, allowPrompt: false });
        runPythonCLI(['status'], setup.pythonCmd);
    });

addCommonOptions(
    program
        .command('set <key> <value>')
)
    .description('设置配置项')
    .action(async function (key, value) {
        const opts = { ...program.opts(), ...this.opts() };
        const setup = await ensureSetup({ ...opts, allowPrompt: false });
        runPythonCLI(['set', key, value], setup.pythonCmd);
    });

addCommonOptions(
    program
        .command('setup')
)
    .description('初始化环境（创建虚拟环境并安装依赖）')
    .action(async function () {
        const opts = { ...program.opts(), ...this.opts() };
        if (opts.useSystemPython) {
            console.log(chalk.yellow('setup 仅用于虚拟环境，系统 Python 请自行安装依赖。'));
            return;
        }
        const pythonCmd = findPython();
        if (!pythonCmd) {
            console.error(chalk.red('Python 3.9+ not found!'));
            process.exit(1);
        }

        console.log(chalk.cyan('Setting up GPU Worker environment...\n'));
        console.log(chalk.green(`Python: ${pythonCmd}`));

        await createVenv(pythonCmd);
        if (!opts.skipInstall) {
            await installDependencies();
        }

        console.log(chalk.green('\n✓ Setup complete!'));
        console.log(chalk.cyan('\nNext steps:'));
        console.log('  1. Run: gpu-worker configure');
        console.log('  2. Run: gpu-worker start');
    });

// 快速启动命令 (无参数时的默认行为)
addCommonOptions(
    program
        .command('quick', { isDefault: true, hidden: true })
)
    .action(async function () {
        const opts = { ...program.opts(), ...this.opts() };
        const setup = await ensureSetup({ ...opts, allowPrompt: true });

        console.log(chalk.cyan.bold('\n  GPU Worker - 分布式GPU推理节点\n'));

        const choices = [
            { name: '🚀 启动 Worker', value: 'start' },
            { name: '⚙️  配置向导', value: 'configure' },
            { name: '📊 查看状态', value: 'status' },
            { name: '📦 安装依赖', value: 'install' },
            { name: '❌ 退出', value: 'exit' }
        ];

        const { action } = await inquirer.prompt([{
            type: 'list',
            name: 'action',
            message: '请选择操作:',
            choices
        }]);

        if (action === 'exit') {
            process.exit(0);
        }

        if (action === 'start') {
            const configPath = path.join(process.cwd(), 'config.yaml');
            if (!fs.existsSync(configPath)) {
                console.log(chalk.yellow('\n未找到配置文件，先进行配置...\n'));
                runPythonCLI(['configure'], setup.pythonCmd);
                return;
            }
        }

        runPythonCLI([action], setup.pythonCmd);
    });

program.parse();
