/**
 * Pyodide Runner - Execute Python code in the browser
 * Provides interactive code execution for ML101
 */

let pyodide;
let pyodideReady = false;
let pyodideLoading = false;

// Initialize Pyodide with required packages
async function initPyodide() {
  if (pyodideReady) {
    return pyodide;
  }

  if (pyodideLoading) {
    // Wait for existing load to complete
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

    // Load essential packages for ML
    console.log('Loading Python packages...');
    await pyodide.loadPackage(['numpy', 'matplotlib', 'scikit-learn', 'scipy']);

    // Setup matplotlib for inline plotting
    pyodide.runPython(`
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('AGG')
import io, base64

def _get_plot_as_base64():
    """Helper to get current plot as base64 string."""
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
    // Show loading state
    buttonElement.textContent = '⏳ Loading Python...';
    buttonElement.disabled = true;
    outputElement.innerHTML = '';
    outputElement.className = 'code-output';
    outputElement.style.display = 'block';

    // Initialize if needed
    const pyodideInstance = await initPyodide();
    
    buttonElement.textContent = '⏳ Running...';

    // Redirect stdout and stderr
    pyodideInstance.runPython(`
import sys
from io import StringIO
sys.stdout = StringIO()
sys.stderr = StringIO()
    `);

    // Run user code
    await pyodideInstance.runPythonAsync(code);

    // Get text output
    const stdout = pyodideInstance.runPython('sys.stdout.getvalue()');
    const stderr = pyodideInstance.runPython('sys.stderr.getvalue()');

    // Check for plots
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
      // Display plot
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

// Add run buttons to interactive code blocks
function addRunButtons() {
  // Find all python-interactive containers
  document.querySelectorAll('.python-interactive').forEach(container => {
    // Don't process twice
    if (container.querySelector('.run-button')) {
      return;
    }

    // Find the code element inside
    const code = container.querySelector('pre code');
    if (!code) return;

    const pre = code.closest('pre');
    if (!pre) return;

    // Get the original code text
    const originalCode = code.textContent;

    // Create editable textarea
    const textarea = document.createElement('textarea');
    textarea.value = originalCode;
    textarea.className = 'code-editor';
    textarea.spellcheck = false;
    textarea.setAttribute('autocapitalize', 'off');
    textarea.setAttribute('autocomplete', 'off');
    textarea.setAttribute('autocorrect', 'off');
    
    // Auto-resize based on content
    const lineCount = originalCode.split('\n').length;
    textarea.rows = Math.max(lineCount, 3);
    
    // Create run button
    const button = document.createElement('button');
    button.textContent = '▶ Run Code';
    button.className = 'run-button md-button md-button--primary';
    button.title = 'Execute this code in your browser';

    // Create reset button
    const resetButton = document.createElement('button');
    resetButton.textContent = '↺ Reset';
    resetButton.className = 'reset-button md-button';
    resetButton.title = 'Reset to original code';
    resetButton.onclick = () => {
      textarea.value = originalCode;
      // Trigger resize
      textarea.rows = Math.max(originalCode.split('\n').length, 3);
    };

    // Create button container
    const buttonContainer = document.createElement('div');
    buttonContainer.className = 'code-buttons';
    buttonContainer.appendChild(button);
    buttonContainer.appendChild(resetButton);

    // Create output area
    const output = document.createElement('div');
    output.className = 'code-output';
    output.style.display = 'none';

    // Add click handler for run button
    button.onclick = async () => {
      const codeText = textarea.value;
      await runPythonCode(codeText, output, button);
    };

    // Handle tab key in textarea
    textarea.addEventListener('keydown', (e) => {
      if (e.key === 'Tab') {
        e.preventDefault();
        const start = textarea.selectionStart;
        const end = textarea.selectionEnd;
        textarea.value = textarea.value.substring(0, start) + '    ' + textarea.value.substring(end);
        textarea.selectionStart = textarea.selectionEnd = start + 4;
      }
      // Ctrl/Cmd + Enter to run
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        button.click();
      }
    });

    // Auto-resize textarea on input
    textarea.addEventListener('input', () => {
      const lines = textarea.value.split('\n').length;
      textarea.rows = Math.max(lines, 3);
    });

    // Hide the original code block and insert our editable version
    const highlightDiv = pre.parentElement;
    highlightDiv.style.display = 'none';
    
    highlightDiv.after(textarea);
    textarea.after(buttonContainer);
    buttonContainer.after(output);
  });
}

// Initialize when DOM is ready
function initialize() {
  addRunButtons();
  
  // Material for MkDocs uses instant navigation
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

// Observe for dynamic content changes
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