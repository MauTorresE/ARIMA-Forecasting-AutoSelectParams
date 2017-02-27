ARIMA forecasting model.

Uses Dickey-Fuller Test to select best model and parameters for the model.

Takes a file path with a file in json format, or a string with json structure.

Sample json:

{
    "fecha": 1467349200000,
    "valor": 44503.07
  }

Returns json (fecha, prediccion, error), error_prom, accuracy