from flask import Flask, render_template, request, send_from_directory
import os
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

disease_mapping = {
    0: "Banded Chlorosis",
    1: "Brown Spot",
    2: "Brown Rust",
    3: "Grassy Shoot Disease",
    4: "Yellow Leaf",
    5: "BacterialBlight",
    6: "Blast",
    7: "BrownSpot (Rice)",
    8: "LeafSmut",
    9: "Tungro"
}

valid_diseases = list(disease_mapping.values())

# Load the trained model
RICE_MODEL_PATH = "model/rice_50epoch_cnn.h5"
SUGARCANE_MODEL_PATH = "model/cnn.h5"
rice_model = load_model(RICE_MODEL_PATH)
sugarcane_model = load_model(SUGARCANE_MODEL_PATH)

# Mapping of class indices to disease names
disease_labels = {
    0: "Banded Chlorosis",
    1: "Brown Spot",
    2: "Brown Rust",
    3: "Grassy Shoot Disease",
    4: "Yellow Leaf",
    5: "Healthy Leaves",
    6: "Dried Leaves",
    7: "Bacterial Blight",
    8: "Blast",
    9: "Brown Spot (Rice)",
    10: "Leaf Smut",
    11: "Tungro"
}


# Disease information
disease_info = {
    "Banded Chlorosis": {
        "Type": "Physiological Disorder / Nutritional Deficiency",
        "Symptoms": [
            "Yellowish-green bands on leaves",
            "Chlorotic (yellowing) areas alternate with green sections",
            "Poor growth and stunted development"
        ],
        "Causes": [
            "Magnesium (Mg) or iron (Fe) deficiency",
            "Poor soil conditions or nutrient imbalance",
            "Drought stress or excessive moisture affecting nutrient uptake"
        ],
        "Treatment & Prevention": [
            "Apply magnesium sulfate or iron chelates as foliar spray",
            "Maintain proper soil pH and fertility through balanced fertilization",
            "Improve soil drainage to prevent excessive moisture buildup",
            "Conduct regular soil testing to monitor nutrient levels"
        ]
    },
    "Brown Spot": {
        "Type": "Fungal Disease (Cercospora longipes)",
        "Symptoms": [
            "Small, reddish-brown circular or oval spots on leaves",
            "Spots may merge, leading to large, necrotic (dead) patches",
            "Premature leaf drying and reduced photosynthesis"
        ],
        "Causes": [
            "High humidity and prolonged leaf wetness",
            "Infection spreads through windborne spores and contaminated plant debris",
            "Poor field sanitation and overuse of nitrogen fertilizers"
        ],
        "Treatment & Prevention": [
            "Use resistant sugarcane varieties",
            "Improve air circulation through proper plant spacing",
            "Avoid excessive nitrogen fertilization",
            "Remove and destroy infected plant debris",
            "Apply fungicides (e.g., copper-based or mancozeb fungicides) if infection spreads"
        ]
    },
    "Brown Rust": {
        "Type": "Fungal Disease (Puccinia melanocephala)",
        "Symptoms": [
            "Small, elongated reddish-brown pustules on leaf surfaces",
            "Severe infections cause leaves to dry and die prematurely",
            "Reduced plant vigor and yield loss"
        ],
        "Causes": [
            "Warm and humid weather conditions",
            "Spores spread through wind, rain, and contaminated farm tools",
            "Susceptible sugarcane varieties are highly affected"
        ],
        "Treatment & Prevention": [
            "Plant rust-resistant sugarcane varieties",
            "Remove and destroy infected leaves to reduce spore load",
            "Avoid excessive nitrogen fertilization, which promotes disease development",
            "Apply fungicides (e.g., propiconazole, triazoles) when necessary"
        ]
    },
    "Grassy Shoot Disease": {
        "Type": "Phytoplasma Disease",
        "Symptoms": [
            "Thin, pale green leaves with excessive tillering (many shoots)",
            "Lack of proper cane formation",
            "Stunted growth and bushy appearance"
        ],
        "Causes": [
            "Sugarcane grassy shoot phytoplasma",
            "Transmitted by insects such as leafhoppers",
            "Infected seed sets contribute to disease spread"
        ],
        "Treatment & Prevention": [
            "Use certified disease-free seed sets for planting",
            "Destroy infected plants to prevent further spread",
            "Control insect vectors with appropriate pesticides",
            "Maintain proper field sanitation by removing diseased plants"
        ]
    },
    "Yellow leaf": {
        "Type": "Fungal Disease (Pyricularia grisea)",
        "Symptoms": [
            "Water-soaked lesions on leaves, turning brown with gray centers",
            "Spots merge, causing leaf blight and drying",
            "Reduced plant vigor and cane production"
        ],
        "Causes": [
            "Warm, humid, and wet conditions",
            "High nitrogen levels can increase susceptibility",
            "Infection spreads through spores carried by wind and rain"
        ],
        "Treatment & Prevention": [
            "Plant resistant sugarcane varieties",
            "Avoid excessive nitrogen fertilization",
            "Improve field drainage to reduce moisture accumulation",
            "Apply fungicides (e.g., tricyclazole, azoxystrobin) if necessary"
        ]
    },
    "BacterialBlight": {
        "Type": "Bacterial Disease",
        "Symptoms": [
            "Wilting of seedlings",
            "Yellowing and drying of leaves"
        ],
        "Causes": [
            "Xanthomonas oryzae pv. oryzae"
        ],
        "Treatment & Prevention": [
            "Obtain seeds from reliable sources",
            "Reduce plant injury during transplanting",
            "Regularly weed the field, levees, and irrigation canals to remove alternate hosts",
            "Avoid excessive nitrogen application; use split applications instead",
            "Plow and dry the field after harvest to expose and eliminate bacteria"
        ]
    },
    "Blast": {
        "Type": "Blast Disease",
        "Symptoms": [
            "Small, spindle-shaped spots with brown borders and gray centers",
            "Spots merge, causing drying and death of leaves"
        ],
        "Causes": [
            "Pyricularia oryzae (Cavara) (fungi)",
            "Magnaporthe oryzae (fungi)",
            "Planting of susceptible varieties",
            "Warm and humid weather conditions",
            "Inadequate irrigation water",
            "Excessive nitrogen use"
        ],
        "Treatment & Prevention": [
            "Use resistant rice varieties",
            "Raise seedlings on wet beds",
            "Avoid excessive nitrogen application",
            "Apply fungicides at early heading and panicle emergence if leaf blast is observed",
            "Destroy infected crop residues"
        ]
    },
    "Tungro": {
        "Type": "Viral Disease",
        "Symptoms": [
            "Leaf discoloration",
            "Stunted growth",
            "Reduced tiller numbers",
            "Sterile or partially filled grains"
        ],
        "Causes": [
            "Combination of two viruses transmitted by green leafhoppers: Rice Tungro Spherical Virus and Rice Tungro Bacilliform Virus"
        ],
        "Treatment & Prevention": [
            "Plant resistant varieties",
            "Use locally adapted resistant varieties",
            "Rotate different resistant varieties every two years to prevent resistance breakdown",
            "Use high-quality seeds",
            "Obtain seeds from accredited seed growers",
            "Practice synchronous planting",
            "Prepare land adequately",
            "Properly level fields for better water management",
            "Follow recommended plant spacing"
        ]
    },
    "LeafSmut": {
        "Type": "Fungal Disease (Entyloma oryzae)",
        "Symptoms": [
            "Small black, raised, angular spots on leaves",
            "Oldest leaves most affected",
            "Severe infections cause leaf tips to die"
        ],
        "Causes": [
            "Fungal spores spread via air, water, or infected plant debris",
            "High humidity and wet conditions",
            "Poor air circulation in densely planted fields",
            "Nutrient deficiency (low nitrogen and potassium)",
            "Contaminated seeds"
        ],
        "Treatment & Prevention": [
            "Plant resistant varieties",
            "Practice crop rotation to break disease cycle",
            "Remove and burn infected plant debris",
            "Use certified disease-free seeds and treat with fungicides before planting",
            "Ensure proper plant spacing for better air circulation",
            "Avoid excessive nitrogen fertilizers",
            "Apply fungicides (copper-based, mancozeb, tricyclazole, or propiconazole if necessary)",
            "Manage irrigation to avoid excessive water accumulation"
        ]
    },
    "BrownSpot (Rice)": {
        "Type": "Fungal Disease (Cochliobolus miyabeanus)",
        "Symptoms": [
            "Numerous large spots on leaves, leading to leaf death",
            "Unfilled, spotted, or discolored grains if seeds are infected"
        ],
        "Causes": [
            "High relative humidity (86−100%) and temperatures (16−36°C)",
            "Unflooded or nutrient-deficient soil",
            "Toxic substance accumulation in soil",
            "Infected seeds (fungus can survive for over four years)",
            "Disease spreads via air, infected rice debris, and weeds",
            "Infection most critical during maximum tillering to ripening stages"
        ],
        "Treatment & Prevention": [
            "Regularly monitor soil nutrients",
            "Apply necessary fertilizers; for silicon-deficient soils, use calcium silicate slag",
            "Use resistant rice varieties (consult local agriculture office for recommendations)",
            "Treat seeds with fungicides (e.g., iprodione, propiconazole, azoxystrobin, trifloxystrobin, carbendazim)",
            "Use hot water seed treatment (53−54°C for 10−12 minutes, pre-soaking in cold water for eight hours enhances effectiveness)"
        ]
    }
}

if not os.path.exists('uploads'):
    os.makedirs('uploads')


@app.route('/camera')  # Flask route for camera.html
def camera():
    return render_template('camera.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save uploaded image
            img_path = os.path.join('uploads', file.filename)
            file.save(img_path)

            # Predict disease
            predicted_disease, confidence_score = predict_disease(img_path)

            if predicted_disease == "Invalid Input":
                return render_template(
                    'result.html',
                    disease="Invalid Input",
                    confidence_score=0.0,
                    details={
                        "Type": "N/A",
                        "Symptoms": ["The image does not match rice or sugarcane diseases."],
                        "Causes": ["N/A"],
                        "Treatment & Prevention": ["Please upload a valid image of rice or sugarcane."]
                    },
                    image_url=f"/uploads/{file.filename}"
                )

            # Fetch disease details
            disease_details = disease_info.get(predicted_disease, {
                "Type": "Unknown",
                "Symptoms": ["No information available"],
                "Causes": ["No information available"],
                "Treatment & Prevention": ["No information available"]
            })

            # Generate image URL for displaying the uploaded image
            image_url = f"/uploads/{file.filename}"

            return render_template(
                'result.html',
                disease=predicted_disease,
                confidence_score=confidence_score,
                details=disease_details,
                image_url=image_url
            )
    return render_template('main.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

def predict_disease(img_path):
    # Preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict disease using rice model
    predictions = rice_model.predict(img_array)  # Replace with your actual model
    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence_score = float(predictions[0][predicted_index])

    # Validate prediction (only rice/sugarcane diseases allowed)
    if confidence_score < 0.5 or predicted_index not in disease_mapping:
        return "Invalid Input", 0.0

    # Get predicted disease
    predicted_disease = disease_mapping.get(predicted_index, "Unknown")
    return predicted_disease, confidence_score

if __name__ == '__main__':
    app.run(debug=True)