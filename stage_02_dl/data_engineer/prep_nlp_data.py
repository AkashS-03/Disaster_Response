import os
import pandas as pd
import numpy as np

def generate_nlp_data():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(base_dir, "data", "nlp")
    os.makedirs(out_dir, exist_ok=True)
    
    # --- TRANSPARENCY DOCUMENTATION ---
    # This dataset is completely synthetic. The severity labels are STRICTLY 
    # generated based on the presence of specific keywords mapped in the templates below.
    # The downstream neural network (GRU) will effectively be learning these rigid rules
    # (e.g., 'trapped' -> SEVERE) rather than organically understanding real human distress.
    
    templates = [
        # SEVERE (Keywords: trapped, rescue, drowning, collapsed, submerged)
        ("People are trapped on the roof of the building on {street}.", "SEVERE"),
        ("We need a rescue boat immediately, water is rising fast.", "SEVERE"),
        ("The bridge collapsed near the main highway.", "SEVERE"),
        ("Someone is drowning in the flood waters!", "SEVERE"),
        ("Water has completely submerged the ground floor, families trapped.", "SEVERE"),
        
        # MODERATE (Keywords: knee-deep, entering, blocked, impassable)
        ("Water is knee-deep on {street}, cars are stalling.", "MODERATE"),
        ("The road is completely blocked by water.", "MODERATE"),
        ("Water is entering the basement of the store.", "MODERATE"),
        ("Traffic is completely impassable due to localized flooding.", "MODERATE"),
        
        # LOW (Keywords: drizzle, puddle, fine, power, minor)
        ("There is a large puddle on {street}, but traffic is moving.", "LOW"),
        ("It's just a light drizzle right now, everything is fine.", "LOW"),
        ("Power is out but no flooding observed yet.", "LOW"),
        ("Just reporting some minor debris on the sidewalk.", "LOW")
    ]
    
    streets = ["MG Road", "Linking Road", "Marine Drive", "SV Road", "LBS Marg", "Carter Road"]
    
    data = []
    # Generate 10000 samples for robust training/testing
    for i in range(10000):
        template, label = templates[np.random.randint(0, len(templates))]
        street = streets[np.random.randint(0, len(streets))]
        text = template.replace("{street}", street)
        
        # Add basic text noise
        if np.random.rand() > 0.7:
            text = text.lower()
        if np.random.rand() > 0.8:
            text += " Please hurry."
        if np.random.rand() > 0.9:
            text = "Umm, " + text
            
        data.append({"transcript": text, "severity": label})
        
    df = pd.DataFrame(data)
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    out_path = os.path.join(out_dir, "synthetic_transcripts.csv")
    df.to_csv(out_path, index=False)
    
    print(f"Synthetic NLP dataset generated: {len(df)} records at {out_path}.")
    print("WARNING: Labels are rule-generated based on hardcoded keywords.")

if __name__ == "__main__":
    generate_nlp_data()
