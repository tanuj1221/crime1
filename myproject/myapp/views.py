# myapp/views.py


from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import re
from datetime import datetime
from django.shortcuts import render
def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path)
    
    # Handling mixed date formats
    data['DATE'] = pd.to_datetime(data['DATE'], errors='coerce', dayfirst=True)
    
    # Optionally, filter out rows with NaT in 'DATE' if necessary
    # data = data.dropna(subset=['DATE'])
    
    return data

def find_time(text):
    # Look for the pattern "At HH:MM" in the text
    time_pattern = r"At (\d{2}:\d{2})"
    matches = re.findall(time_pattern, str(text))
    return matches[0] if matches else None


# Plot the distribution of categorized times
def plot_time_distribution(data):
    # First, extract and categorize times from the 'INCIDENT' column or any relevant column
    data['TIME'] = data['INCIDENT'].apply(find_time)  # Assuming 'INCIDENT' is the column to extract from
    data['TIME_OF_DAY'] = data['TIME'].apply(categorize_time)
    
    # Then, plot
    plt.figure(figsize=(10, 6))
    sns.countplot(x=data['TIME_OF_DAY'], order=['Morning', 'Afternoon', 'Evening', 'Night'])
    plt.title('Distribution of Incidents by Time of Day')
    plt.xlabel('Time of Day')
    plt.ylabel('Number of Incidents')
    plt.tight_layout()
    # Convert plot to PNG image
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph

# Categorize time into morning, afternoon, evening, night
def categorize_time(time_str):
    if time_str is None:
        return None
    hour = int(time_str.split(':')[0])
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour <= 23:
        return 'Evening'
    else:
        return 'Night'

def generate_heatmap_html(file_path):
    # Load the dataset
    data = load_and_prepare_data(file_path)
    
    # Filter out rows where coordinates are missing
    data.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    
    # Creating a map centered at the average location
    map_center = [data['Latitude'].mean(), data['Longitude'].mean()]
    folium_map = folium.Map(location=map_center, zoom_start=6)
    
    # Adding the HeatMap layer to the map
    HeatMap(data[['Latitude', 'Longitude']].values.tolist()).add_to(folium_map)
    
    # Instead of saving, return the HTML embed code
    return folium_map._repr_html_()

def plot_day_distribution(data):
    """Generate a plot for incidents by day of the week."""
    # Extract day names
    day_names = data['DATE'].dt.day_name()
    # Order for sorting days of the week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # Creating a categorical type with the specified order
    day_names = pd.Categorical(day_names, categories=day_order, ordered=True)
    # Plot
    plt.figure(figsize=(10, 6))
    sns.countplot(x=day_names)
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Convert plot to PNG image
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph


def plot_area_distribution(data):
    """Generate a plot for incidents by area."""
    plt.figure(figsize=(12, 8))
    area_counts = data['AREA'].value_counts().head(10)  # Top 10 areas
    sns.barplot(x=area_counts.values, y=area_counts.index)
    plt.title('Top 10 Areas by Number of Incidents')
    plt.xlabel('Number of Incidents')
    plt.ylabel('Area')
    plt.tight_layout()
    # Convert plot to PNG image
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph

# Function to categorize time into 3-hour intervals
def categorize_into_intervals(time_str):
    if time_str is None or time_str == "Unknown":
        return "Unknown"
    hour = int(time_str.split(':')[0])  # Ensure the string is in a valid time format
    if 0 <= hour < 3:
        return '00:00-02:59'
    elif 3 <= hour < 6:
        return '03:00-05:59'
    elif 6 <= hour < 9:
        return '06:00-08:59'
    elif 9 <= hour < 12:
        return '09:00-11:59'
    elif 12 <= hour < 15:
        return '12:00-14:59'
    elif 15 <= hour < 18:
        return '15:00-17:59'
    elif 18 <= hour < 21:
        return '18:00-20:59'
    else:
        return '21:00-23:59'

def generate_prediction_heatmap(prediction_data):
    # Creating a map centered at the average location
    map_center = [prediction_data['Latitude'].mean(), prediction_data['Longitude'].mean()]
    folium_map = folium.Map(location=map_center, zoom_start=6)
    
    # Adding the HeatMap layer to the map
    HeatMap(prediction_data[['Latitude', 'Longitude', 'PREDICTED_INCIDENTS']].values.tolist()).add_to(folium_map)
    
    return folium_map._repr_html_()
# Plot the distribution of incidents in 3-hour intervals
def plot_time_intervals_distribution(data):
    data['TIME'] = data['INCIDENT'].apply(find_time)  # Assuming 'INCIDENT' is your text column
    data['TIME_INTERVAL'] = data['TIME'].apply(categorize_into_intervals)
    
    plt.figure(figsize=(12, 8))
    interval_order = ['00:00-02:59', '03:00-05:59', '06:00-08:59', '09:00-11:59',
                      '12:00-14:59', '15:00-17:59', '18:00-20:59', '21:00-23:59']
    sns.countplot(y=data['TIME_INTERVAL'], order=interval_order)
    plt.title('Distribution of Incidents by 3-Hour Intervals')
    plt.xlabel('Number of Incidents')
    plt.ylabel('Time Interval')
    plt.tight_layout()
    
    # Convert plot to PNG image
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph
label_encoder_area = LabelEncoder()
label_encoder_time_interval = LabelEncoder()

def index(request):
    file_path = "./updated_main.csv"  # Update as necessary
    data = load_and_prepare_data(file_path)
    daily_incident_counts = data.groupby(['DATE', 'AREA']).size().reset_index(name='DAILY_INCIDENT_COUNT')
    data = pd.merge(data, daily_incident_counts, on=['DATE', 'AREA'], how='left')
    
    # Generate visualizations
    area_distribution_img = plot_area_distribution(data)
    day_distribution_img = plot_day_distribution(data)
    time_distribution_img = plot_time_distribution(data)
    time_intervals_distribution_img = plot_time_intervals_distribution(data)
    
    label_encoder_area = LabelEncoder()
    label_encoder_time_interval = LabelEncoder()
    data['AREA_ENCODED'] = label_encoder_area.fit_transform(data['AREA'])
    data['TIME_INTERVAL_ENCODED'] = label_encoder_time_interval.fit_transform(data['TIME_INTERVAL'])

    data['DATE'] = data['DATE'].dt.dayofyear
    data = data.fillna(0)
    areas = data['AREA'].unique()

    
    context = {
        'area_distribution_img': day_distribution_img,
        'day_distribution_img': area_distribution_img,
        'time_distribution_img': time_distribution_img,
        'time_intervals_distribution_img': time_intervals_distribution_img,
        'areas': areas,
       
    }
        
    if request.method == "POST":
        # Process form data
        date = request.POST.get("date")
        time_interval = request.POST.get("time_interval")
        
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        day_of_year = date_obj.timetuple().tm_yday
            # Process form data
        date = request.POST.get("date")
        area = request.POST.get("area")
        time_interval = request.POST.get("time_interval")
        
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        day_of_year = date_obj.timetuple().tm_yday

        # Transform area and time_interval using the fitted LabelEncoders
        area_encoded = label_encoder_area.transform([area])[0]
        time_interval_encoded = label_encoder_time_interval.transform([time_interval])[0]
        
        # Prepare the input for prediction
        prediction_input = [[day_of_year, area_encoded, time_interval_encoded]]

        # Prepare the input for prediction
        areas = data['AREA'].unique()
        prediction_input = []
        for area in areas:
            area_encoded = label_encoder_area.transform([area])[0] if area in label_encoder_area.classes_ else -1
            time_interval_encoded = label_encoder_time_interval.transform([time_interval])[0] if time_interval in label_encoder_time_interval.classes_ else -1
            prediction_input.append([day_of_year, area_encoded, time_interval_encoded])
        
        # Train the model
        X = data[['DATE', 'AREA_ENCODED', 'TIME_INTERVAL_ENCODED']]
        y = data['DAILY_INCIDENT_COUNT']
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions for all areas
        predictions = model.predict(prediction_input)
        prediction_result = model.predict(prediction_input)[0]
        context.update({'prediction_result':f"crime prediction on {area} is {prediction_result} during {time_interval} on {date}"})

        
        # Create a DataFrame with predicted incidents for each area
        prediction_data = pd.DataFrame({'AREA': areas, 'PREDICTED_INCIDENTS': predictions})
        
        # Merge with the original data to get the coordinates
        prediction_data = pd.merge(prediction_data, data[['AREA', 'Latitude', 'Longitude']].drop_duplicates(), on='AREA', how='left')
        
        # Generate the heatmap
        heatmap_html = generate_prediction_heatmap(prediction_data)
        
        context['heatmap_html'] = heatmap_html
    
    return render(request, 'myapp/index.html', context)


