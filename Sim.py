from forecasting import torchLSTM as tl
from PythonDataProcessing import DataRetrieval as DR
import datetime
import time


while True:
    hour = datetime.datetime.now().hour
    minute = datetime.datetime.now().minute
    if hour == 8 and minute == 0:
        print(datetime.datetime.now())
        #Getting top predicted currencies
        stock_lst = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'CMCSA', 'JPM', 'HD', 'DIS', 'XOM']
        rtrn_lst = []
        for stock in stock_lst:
            try:
                key = stock
                predictor = predictor = tl.PredictionEngine(key,is_loading=True,training_set_coeff=1,period='max',interval='1d',normalizer_type='Relative') #Creates model
                pos,pred_now,pred_prev = predictor.predict_now() # Get's prediction for tomorrow
                close_price = DR.get_last_stock_price(key,'1d')

                print(key)
                print("Price moving up? " + str(pos))
                print("Next prediction: " + str(pred_now))
                print("Previous Prediction: " + str(pred_prev))

                rtrn_lst.append([key,pos,pred_now,pred_prev,close_price])
            except:
                print("ERRRRRRRORRRRRRRR")

        # Might want to sort according to predicted return.
        # That would involve moving code from below to the loop above.
        rtrn_lst = sorted(rtrn_lst,key=lambda l:l[1], reverse=True)
        for element in rtrn_lst:

            print(element[0])
            print("Price moving up?: " + str(element[1])) # Is price moving up today?
            print("Next prediction: " + str(element[2])) # Predicted price for todays close
            print("Previous Prediction: " + str(element[3])) # Predicted price for yesterdays close
            pred_return = element[2]-element[3] / element[3] # Predicted return by end of day.
            print("Predicted Return: " + str(pred_return))
            print("Close Price: " + str(element[4])) # Actual close value from yesterday.

            # Element[1] is the variable pos.
            # This indicates if price is moving up for the day.
            # If it is true, then the price is moving up.
            if element[1]:
                print("BUY")
        time.sleep(60)
