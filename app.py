import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model_xgb.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    
    '''
    
    #int_features = [int(x) for x in request.form.values()]
    
    features = list(request.form.values())
    Beds=features[0]
    Baths=features[1]
    Parking=features[2]
    Sqft=features[3]
    City=features[4]
    Numeric_cols=['Sqft', 'Beds', 'Bath', 'Parking']
    City_Names=[ 'Ajax', 'Aurora', 'Barrie','Belleville', 'Brampton', 'BrantCounty', 'Brantford', 'BruceCounty','Burlington', 'Cambridge', 'Chatham-Kent', 'CochraneDistrict',
       'Collingwood', 'DufferinCounty', 'Dunnville', 'DurhamRegion','EastYork', 'ElginCounty', 'EssexCounty', 'Etobicoke', 'FortErie','FrontenacCounty', 'Georgetown', 'GreyCounty', 'Grimsby', 'Guelph','HaldimandCounty', 'HaliburtonCounty', 'HaltonRegion', 'Hamilton','HastingsCounty', 'HuronCounty', 'KawarthaLakes', 'Kingston','Kitchener', 'Kleinburg', 'LambtonCounty', 'LanarkCounty','LeedsandGrenvilleCounties', 'LennoxandAddingtonCounty', 'London','Maple', 'Markham', 'MiddlesexCounty', 'Milton', 'Mississauga',
       'MuskokaDistrict', 'Newmarket', 'NiagaraFalls', 'NiagaraRegion','NipissingDistrict', 'NorfolkCounty', 'NorthYork','NorthumberlandCounty', 'Oakville', 'Orangeville', 'Orillia', 'Oshawa','Ottawa', 'OxfordCounty', 'ParrySoundDistrict', 'PeelRegion','PerthCounty', 'Peterborough', 'PeterboroughCounty', 'PortColborne','PrinceEdwardCounty', 'QuinteWest', 'RegionofWaterloo', 'RichmondHill','Scarborough', 'SimcoeCounty', 'St.Catharines', 'St.Thomas','StoneyCreek', 'Stratford', 'SudburyDistrict', 'Thornhill', 'Thorold','ThunderBayDistrict', 'Toronto', 'Unionville', 'Vaughan', 'WasagaBeach','Waterloo', 'Welland', 'WellingtonCounty', 'Whitby','Whitchurch-Stouffville', 'Woodbridge', 'Woodstock', 'York','YorkRegion']
    Column_Names= Numeric_cols+City_Names
    df=pd.DataFrame(columns=Column_Names)
    df.at[0,'Sqft']=Sqft
    df.at[0,'Beds']=Beds
    df.at[0,'Bath']=Baths
    df.at[0,'Parking']=Parking
    for col in df.columns:
        if col == City:
            df.at[0,col]=1
    df=df.fillna(0)
    
          
    
    
    
    print("My features:",features)
    
    #final_features = [np.array(int_features)]
    prediction = model.predict(df.values)

    output = int(prediction)

    return render_template('index.html', prediction_text='Price of House will be $ {}'.format(output), Sqft = Sqft, Beds = Beds, Baths = Baths, Parking = Parking, City = City)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)