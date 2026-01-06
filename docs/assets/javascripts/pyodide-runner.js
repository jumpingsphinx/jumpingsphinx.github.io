/**
 * Pyodide Runner - Execute Python code in the browser
 * Provides interactive code execution for ML101
 */

let pyodide;
let pyodideReady = false;

// Initialize Pyodide with required packages
async function initPyodide() {
  if (pyodideReady) {
    return pyodide;
  }

  if (!pyodide) {
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
    `);

    pyodideReady = true;
    console.log('Pyodide ready!');
  }

  return pyodide;
}

// Run Python code and capture output
async function runPythonCode(code, outputElement, buttonElement) {
  const originalText = buttonElement.textContent;

  try {
    // Show loading state
    buttonElement.textContent = '⏳ Running...';
    buttonElement.disabled = true;
    outputElement.textContent = '';
    outputElement.className = 'code-output';

    // Initialize if needed
    const pyodideInstance = await initPyodide();

    // Redirect stdout and stderr
    pyodideInstance.runPython(`
      import sys
      from io import StringIO
      sys.stdout = StringIO()
      sys.stderr = StringIO()
      _plot_counter = 0
    `);

    // Run user code
    await pyodideInstance.runPythonAsync(code);

    // Get text output
    const stdout = pyodideInstance.runPython('sys.stdout.getvalue()');
    const stderr = pyodideInstance.runPython('sys.stderr.getvalue()');

    // Check for plots
    const hasPlot = pyodideInstance.runPython(`
      import matplotlib.pyplot as plt
      bool(plt.get_fignums())
    `);

    let output = '';
    if (stdout) output += stdout;
    if (stderr) output += stderr;

    if (hasPlot) {
      // Get plot as base64 image
      const plotData = pyodideInstance.runPython(`
        import matplotlib.pyplot as plt
        import io, base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close('all')
        img_str
      `);

      // Display plot
      const img = document.createElement('img');
      img.src = 'data:image/png;base64,' + plotData;
      img.style.maxWidth = '100%';
      img.style.height = 'auto';
      img.style.marginTop = '10px';
      outputElement.appendChild(img);
    }

    if (output.trim()) {
      const pre = document.createElement('pre');
      pre.textContent = output;
      pre.style.margin = hasPlot ? '10px 0 0 0' : '0';
      outputElement.insertBefore(pre, outputElement.firstChild);
    } else if (!hasPlot) {
      outputElement.textContent = '✓ Code executed successfully (no output)';
    }

  } catch (error) {
    outputElement.className = 'code-output error';
    outputElement.textContent = `Error: ${error.message}`;
  } finally {
    buttonElement.textContent = originalText;
    buttonElement.disabled = false;
  }
}

// Add run buttons to interactive code blocks
function addRunButtons() {
  // Find all python-interactive containers and add buttons
  document.querySelectorAll('.python-interactive').forEach(container => {
    // Don't add button twice
    if (container.querySelector('.run-button')) {
      return;
    }

    // Find the code element inside
    const code = container.querySelector('pre code');
    if (!code) return;

    const pre = code.closest('pre');
    if (!pre) return;

    // Create run button
    const button = document.createElement('button');
    button.textContent = '▶ Run Code';
    button.className = 'run-button md-button md-button--primary';
    button.title = 'Execute this code in your browser';

    // Create output area
    const output = document.createElement('div');
    output.className = 'code-output';

    // Add click handler
    button.onclick = async () => {
      const codeText = code.textContent;
      await runPythonCode(codeText, output, button);
    };

    // Insert button and output after the highlight div
    const highlightDiv = pre.parentElement;
    highlightDiv.after(button);
    button.after(output);
  });
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', addRunButtons);
} else {
  addRunButtons();
}

// Also run when navigating in single-page mode
document.addEventListener('DOMContentLoaded', () => {
  // Material for MkDocs uses instant navigation
  const observer = new MutationObserver(() => {
    addRunButtons();
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
});
