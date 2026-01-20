#!/usr/bin/env node
/**
 * Post-install script
 * 在 npm install 后自动设置 Python 环境
 */

const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const PACKAGE_DIR = path.resolve(__dirname, '..');

// ANSI colors (不依赖 chalk)
const colors = {
    green: (s) => `\x1b[32m${s}\x1b[0m`,
    yellow: (s) => `\x1b[33m${s}\x1b[0m`,
    cyan: (s) => `\x1b[36m${s}\x1b[0m`,
    red: (s) => `\x1b[31m${s}\x1b[0m`,
    bold: (s) => `\x1b[1m${s}\x1b[0m`
};

console.log(colors.cyan('\n📦 GPU Worker - Post-install setup\n'));

// 检测 Python
function findPython() {
    const pythonCommands = ['python3', 'python', 'py'];

    for (const cmd of pythonCommands) {
        try {
            const version = execSync(`${cmd} --version`, {
                encoding: 'utf8',
                stdio: ['pipe', 'pipe', 'pipe']
            });
            const match = version.match(/Python (\d+)\.(\d+)/);
            if (match && parseInt(match[1]) >= 3 && parseInt(match[2]) >= 9) {
                return { cmd, version: version.trim() };
            }
        } catch (e) {
            continue;
        }
    }
    return null;
}

const python = findPython();

if (python) {
    console.log(colors.green(`✓ Found ${python.version}`));
    console.log(colors.cyan('\nRun the following to get started:'));
    console.log(colors.bold('  npx gpu-worker configure'));
    console.log(colors.bold('  npx gpu-worker start'));
} else {
    console.log(colors.yellow('⚠ Python 3.9+ not found'));
    console.log(colors.yellow('\nPlease install Python 3.9+ before using gpu-worker:'));
    console.log('  - Windows: https://www.python.org/downloads/');
    console.log('  - macOS:   brew install python@3.11');
    console.log('  - Ubuntu:  sudo apt install python3.11 python3.11-venv');
}

console.log(colors.cyan('\nFor more info: npx gpu-worker --help\n'));
