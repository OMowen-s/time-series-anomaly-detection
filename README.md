Time Series Anomaly Detection System


📌 Project Overview
This project implements a robust anomaly detection system for time series data using the Interquartile Range (IQR) method. It automates the detection of outliers, generates detailed validation and anomaly reports, cleans the data, and visualizes anomalies with annotated plots.

Key Features:

Anomaly Detection: Identifies anomalies in time series data using the IQR method.

Validation Reports: Generates comprehensive reports detailing the anomalies detected.

Data Cleaning: Provides cleaned data by removing or correcting anomalies.

Visualization: Creates annotated plots highlighting anomalies for easy interpretation.​

⚙️ Installation
Prerequisites
Ensure you have the following installed:

Python 3.7 or higher

pip (Python package installer)​
codefinity.com

Steps
Clone the Repository:
git clone https://github.com/yourusername/anomaly-detection.git
cd anomaly-detection
Create a Virtual Environment (Optional but Recommended):
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install Required Packages:
pip install -r requirements.txt

🛠️ Usage
Running the Anomaly Detection
python detect_anomalies.py --input data/input.csv --output reports/
Arguments:

--input: Path to the input CSV file containing time series data.

--output: Directory to save the generated reports and cleaned data.​

Example

python detect_anomalies.py --input data/temperature.csv --output reports/
This command will process temperature.csv, detect anomalies, and save the results in the reports/ directory.​

📄 Report Outputs
After running the detection script, the following files will be generated in the specified output directory:

validation_report.csv: Lists all detected anomalies with timestamps and values.

anomaly_positions.csv: Provides the indices and corresponding values of anomalies.

cleaned_data.csv: Contains the original data with anomalies removed or corrected.

anomaly_plot.png: A plot visualizing the time series data with anomalies highlighted.​

📊 Visualization
The system generates a plot (anomaly_plot.png) that visualizes the time series data with anomalies marked. Anomalies are highlighted in red, and the plot includes dashed lines indicating the upper and lower bounds based on the IQR method.​
Medium

🔧 Customization
You can adjust the sensitivity of the anomaly detection by modifying the iqr_factor parameter in the detect_anomalies.py script:

iqr_factor = 1.5  # Default value
Increasing this value will make the detection more sensitive (detecting more anomalies), while decreasing it will make it less sensitive.​

🧪 Testing
To run the test suite:

Ensure that your test data is placed in the tests/data/ directory.​

🧑‍💻 Contributing
Contributions are welcome! To contribute:

Fork the repository.

Create a new branch for your feature or bugfix.

Make your changes.

Run tests to ensure everything is working.

Submit a pull request.​
s-ai-f.github.io

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
