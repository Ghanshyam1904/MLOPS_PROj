from flask import Flask , request , render_template
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')

    else:
        try:
            # Validate required fields
            required_fields = ['carat', 'depth', 'table', 'x', 'y', 'z', 'cut', 'color', 'clarity']
            for field in required_fields:
                if not request.form.get(field):
                    return render_template('form.html', error=f'Please fill in the {field} field.')

            data = CustomData(
                carat = float(request.form.get('carat')),
                depth = float(request.form.get('depth')),
                table = float(request.form.get('table')),
                x = float(request.form.get('x')),
                y = float(request.form.get('y')),
                z = float(request.form.get('z')),
                cut = request.form.get('cut'),
                color = request.form.get('color'),
                clarity = request.form.get('clarity')
            )

            final_new_data = data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predicted(final_new_data)

            results = round(pred[0],2)

            return render_template('form.html', final_result=results)

        except ValueError as e:
            return render_template('form.html', error='Please enter valid numeric values for all measurements.')
        except Exception as e:
            return render_template('form.html', error=f'An error occurred: {str(e)}')


if __name__ == '__main__':
       app.run(host='0.0.0.0', debug=False)
