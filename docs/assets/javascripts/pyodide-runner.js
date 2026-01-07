/**
 * Pyodide Runner - Execute Python code in the browser
 * Provides interactive code execution for ML101 with CodeMirror editor
 */

let pyodide;
let pyodideReady = false;
let pyodideLoading = false;
let codeMirrorEditors = new Map();

// Initialize Pyodide with required packages
async function initPyodide() {
  if (pyodideReady) {
    return pyodide;
  }

  if (pyodideLoading) {
    while (!pyodideReady) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    return pyodide;
  }

  pyodideLoading = true;

  try {
    console.log('Loading Pyodide...');
    pyodide = await loadPyodide({
      indexURL: "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/"
    });

    console.log('Loading Python packages...');
    await pyodide.loadPackage(['numpy', 'matplotlib', 'scikit-learn', 'scipy']);

    pyodide.runPython(`
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('AGG')
import io, base64

def _get_plot_as_base64():
    if not plt.get_fignums():
        return None
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='white')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close('all')
    return img_str
    `);

    pyodideReady = true;
    console.log('Pyodide ready!');
  } catch (error) {
    console.error('Failed to initialize Pyodide:', error);
    pyodideLoading = false;
    throw error;
  }

  return pyodide;
}

// Run Python code and capture output
async function runPythonCode(code, outputElement, buttonElement) {
  const originalText = buttonElement.textContent;

  try {
    buttonElement.textContent = '⏳ Loading Python...';
    buttonElement.disabled = true;
    outputElement.innerHTML = '';
    outputElement.className = 'code-output';
    outputElement.style.display = 'block';

    const pyodideInstance = await initPyodide();
    
    buttonElement.textContent = '⏳ Running...';

    pyodideInstance.runPython(`
import sys
from io import StringIO
sys.stdout = StringIO()
sys.stderr = StringIO()
    `);

    await pyodideInstance.runPythonAsync(code);

    const stdout = pyodideInstance.runPython('sys.stdout.getvalue()');
    const stderr = pyodideInstance.runPython('sys.stderr.getvalue()');
    const plotData = pyodideInstance.runPython('_get_plot_as_base64()');

    let output = '';
    if (stdout) output += stdout;
    if (stderr) output += stderr;

    if (output.trim()) {
      const pre = document.createElement('pre');
      pre.textContent = output;
      pre.style.margin = '0';
      pre.style.whiteSpace = 'pre-wrap';
      pre.style.wordWrap = 'break-word';
      outputElement.appendChild(pre);
    }

    if (plotData) {
      const img = document.createElement('img');
      img.src = 'data:image/png;base64,' + plotData;
      img.style.maxWidth = '100%';
      img.style.height = 'auto';
      img.style.marginTop = output.trim() ? '10px' : '0';
      img.style.borderRadius = '4px';
      outputElement.appendChild(img);
    }

    if (!output.trim() && !plotData) {
      outputElement.innerHTML = '<pre style="margin:0;color:#4caf50;">✓ Code executed successfully (no output)</pre>';
    }

  } catch (error) {
    outputElement.className = 'code-output error';
    outputElement.innerHTML = `<pre style="margin:0;">Error: ${error.message}</pre>`;
  } finally {
    buttonElement.textContent = originalText;
    buttonElement.disabled = false;
  }
}

// Add CodeMirror editors to interactive code blocks
function addRunButtons() {
  document.querySelectorAll('.python-interactive').forEach((container, index) => {
    if (container.querySelector('.code-editor-wrapper')) {
      return;
    }

    const code = container.querySelector('pre code');
    if (!code) return;

    const pre = code.closest('pre');
    if (!pre) return;

    const originalCode = code.textContent;

    // Create wrapper for CodeMirror
    const editorWrapper = document.createElement('div');
    editorWrapper.className = 'code-editor-wrapper';
    
    // Create the editor container
    const editorContainer = document.createElement('div');
    editorContainer.className = 'code-editor-container';
    editorContainer.id = `code-editor-${index}`;
    editorWrapper.appendChild(editorContainer);

    // Create button container
    const buttonContainer = document.createElement('div');
    buttonContainer.className = 'code-buttons';

    // Create run button
    const button = document.createElement('button');
    button.textContent = '▶ Run Code';
    button.className = 'run-button md-button md-button--primary';
    button.title = 'Execute this code (Ctrl+Enter)';

    // Create reset button
    const resetButton = document.createElement('button');
    resetButton.textContent = '↺ Reset';
    resetButton.className = 'reset-button md-button';
    resetButton.title = 'Reset to original code';

    buttonContainer.appendChild(button);
    buttonContainer.appendChild(resetButton);

    // Create output area
    const output = document.createElement('div');
    output.className = 'code-output';
    output.style.display = 'none';

    // Hide original and insert new elements
    const highlightDiv = pre.parentElement;
    highlightDiv.style.display = 'none';
    
    highlightDiv.after(editorWrapper);
    editorWrapper.after(buttonContainer);
    buttonContainer.after(output);

    // Initialize CodeMirror
    const isDark = document.documentElement.getAttribute('data-md-color-scheme') === 'slate';
    const editor = CodeMirror(editorContainer, {
      value: originalCode,
      mode: 'python',
      theme: isDark ? 'material-darker' : 'default',
      lineNumbers: true,
      indentUnit: 4,
      tabSize: 4,
      indentWithTabs: false,
      lineWrapping: true,
      matchBrackets: true,
      autoCloseBrackets: true,
      extraKeys: {
        'Ctrl-Enter': () => button.click(),
        'Cmd-Enter': () => button.click(),
        'Tab': (cm) => {
          if (cm.somethingSelected()) {
            cm.indentSelection('add');
          } else {
            cm.replaceSelection('    ', 'end');
          }
        }
      }
    });

    // Store editor reference
    codeMirrorEditors.set(container, editor);

    // Button handlers
    button.onclick = async () => {
      const codeText = editor.getValue();
      await runPythonCode(codeText, output, button);
    };

    resetButton.onclick = () => {
      editor.setValue(originalCode);
    };

    // Refresh editor when it becomes visible
    setTimeout(() => editor.refresh(), 100);
  });
}

// Update CodeMirror themes when color scheme changes
function updateEditorThemes() {
  const isDark = document.documentElement.getAttribute('data-md-color-scheme') === 'slate';
  const theme = isDark ? 'material-darker' : 'default';
  
  codeMirrorEditors.forEach(editor => {
    editor.setOption('theme', theme);
  });
}

// Watch for theme changes
const themeObserver = new MutationObserver((mutations) => {
  mutations.forEach((mutation) => {
    if (mutation.attributeName === 'data-md-color-scheme') {
      updateEditorThemes();
    }
  });
});

// Initialize
function initialize() {
  addRunButtons();
  themeObserver.observe(document.documentElement, { attributes: true });
  
  if (typeof document$ !== 'undefined') {
    document$.subscribe(() => {
      setTimeout(addRunButtons, 100);
    });
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initialize);
} else {
  initialize();
}

const observer = new MutationObserver((mutations) => {
  let shouldAddButtons = false;
  for (const mutation of mutations) {
    if (mutation.addedNodes.length > 0) {
      shouldAddButtons = true;
      break;
    }
  }
  if (shouldAddButtons) {
    setTimeout(addRunButtons, 100);
  }
});

observer.observe(document.body, {
  childList: true,
  subtree: true
});