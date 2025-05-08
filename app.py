import os
import json
import traceback
from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
# Corrected import:
from datetime import datetime, timedelta
import matplotlib

# IMPORTANT: Set backend *before* importing pyplot or your script if it imports pyplot
# Also good practice for web servers where display is not available.
matplotlib.use('Agg')

# Import your main function AFTER setting the backend
try:
    # Assuming 4.3success.py is in the same directory as app.py
    from success import run_volatility_prediction, FinancialDataProcessor, VolatilityPredictionModels # Renamed file for clarity
    # You might need to initialize the models class here if it holds state/device info
    # model_builder = VolatilityPredictionModels() # Or handle device setup here if needed
    print("Successfully imported prediction script.")
except ImportError as e:
    print(f"Error importing prediction script: {e}")
    print("Please ensure '4_3success.py' (or your actual script name) is in the same directory as 'app.py'.")
    # Define dummy functions if import fails, so Flask can still start
    def run_volatility_prediction(*args, **kwargs):
        raise NotImplementedError("Prediction script failed to import.")

# --- Flask App Setup ---
app = Flask(__name__)

# Configuration
# Ensure this path is correct relative to where you run app.py
# If output/ is in the same dir as app.py, this is fine.
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'output'))
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, 'visualizations')

app.config['OUTPUT_DIR'] = OUTPUT_DIR
app.config['VISUALIZATIONS_DIR'] = VISUALIZATIONS_DIR

# Ensure output directories exist (optional, your script likely does this)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# --- Helper Function ---
def find_image_url(symbol, target_col, image_type, cache_buster=True):
    """Finds the URL for a specific generated image."""
    filename = None
    if image_type == 'data_analysis':
        # Use the _visualization.png pattern instead of _data_analysis.png
        safe_symbol = symbol.replace('^', '_')
        filename = f"{safe_symbol}_visualization.png"
    elif image_type == 'predictions':
        filename = f"{symbol}_{target_col}_predictions_comparison.png"
    elif image_type == 'rmse':
        filename = f"{symbol}_{target_col}_rmse_comparison.png"
    elif image_type == 'r2':
        filename = f"{symbol}_{target_col}_r2_comparison.png"
    elif image_type == 'mae':
        filename = f"{symbol}_{target_col}_mae_comparison.png"
    elif image_type == 'mse':
        filename = f"{symbol}_{target_col}_mse_comparison.png"

    if filename:
        # Check if file actually exists in the expected location
        full_path = os.path.join(app.config['VISUALIZATIONS_DIR'], filename)
        if os.path.exists(full_path):
            url = url_for('serve_visualization', filename=filename)
            if cache_buster:
                url += f"?t={datetime.now().timestamp()}"
            return url
        else:
            print(f"Image file not found: {full_path}")
            return None
    return None

# --- Routes ---
@app.route('/')
def index():
    """Renders the main page."""
    # Default values or suggestions
    default_symbol = '^GSPC'
    default_window = 21
    # This line now works because timedelta is imported
    default_start = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    default_end = datetime.now().strftime('%Y-%m-%d')
    return render_template('index.html',
                           default_symbol=default_symbol,
                           default_window=default_window,
                           default_start=default_start,
                           default_end=default_end)

@app.route('/run_prediction', methods=['POST'])
def handle_prediction():
    """Handles the prediction request from the frontend."""
    try:
        # Get data from the AJAX request
        data = request.get_json()
        symbol = data.get('symbol', '^GSPC').strip()
        target_window = int(data.get('target_window', 21))
        start_date = data.get('start_date', '2018-01-01')
        end_date = data.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        optimize = bool(data.get('optimize_params', False))
        seq_length = 20 # Or get from form if needed
        test_size = 0.2 # Or get from form if needed

        print(f"Received request: Symbol={symbol}, Window={target_window}, "
              f"Start={start_date}, End={end_date}, Optimize={optimize}")

        # --- Run the core prediction logic ---
        # Make sure the output dir used by the script matches app.config['OUTPUT_DIR']
        # Also ensure the script filename used in the import above matches your file
        results = run_volatility_prediction(
            symbol=symbol,
            target_volatility_window=target_window,
            start_date=start_date,
            end_date=end_date,
            test_size=test_size,
            seq_length=seq_length,
            output_dir=app.config['OUTPUT_DIR'],
            optimize_params=optimize
        )
        # --- End prediction logic ---

        print("Prediction script finished.")
        target_col = results.get('target_col', f'volatility_{target_window}d')
        evaluations = results.get('evaluations', [])

        # Prepare response
        response_data = {
            'status': 'success',
            'message': f'Prediction completed for {symbol} ({target_col}).',
            'evaluations': evaluations,
            'images': {
                'data_analysis': find_image_url(symbol, target_col, 'data_analysis'),
                'predictions': find_image_url(symbol, target_col, 'predictions'),
                'rmse': find_image_url(symbol, target_col, 'rmse'),
                'r2': find_image_url(symbol, target_col, 'r2'),
                'mae': find_image_url(symbol, target_col, 'mae'),
                'mse': find_image_url(symbol, target_col, 'mse')
            },
            'symbol': symbol, # Send back for reference if needed
            'target_col': target_col
        }
        return jsonify(response_data)

    except NotImplementedError as e:
         print(f"Execution Error: {e}")
         return jsonify({'status': 'error', 'message': 'Prediction script not imported correctly.'}), 500
    except Exception as e:
        print(f"Error during prediction: {e}")
        print(traceback.format_exc()) # Print detailed traceback to console
        return jsonify({
            'status': 'error',
            'message': f'An error occurred: {str(e)} Check server logs for details.'
        }), 500

@app.route('/visualizations/<path:filename>')
def serve_visualization(filename):
    """Serves images from the visualization directory."""
    # Security: Basic check to prevent accessing files outside the intended dir
    if '..' in filename or filename.startswith('/'):
         return "Forbidden", 403
    # Make sure VISUALIZATIONS_DIR is correct
    directory = app.config['VISUALIZATIONS_DIR']
    # print(f"Attempting to serve: {filename} from {directory}") # Debugging line
    # Check if file exists before sending
    if not os.path.exists(os.path.join(directory, filename)):
        print(f"File not found for serving: {os.path.join(directory, filename)}")
        return "File not found", 404
    return send_from_directory(directory, filename)


# --- Main Execution ---
if __name__ == '__main__':
    # Set host='0.0.0.0' to make it accessible on your network
    # Use threaded=False if you encounter issues with matplotlib/torch in threads
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=True) # Added threaded=True for potentially better handling