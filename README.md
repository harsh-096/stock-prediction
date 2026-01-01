# NSE Intraday Trading Bot

Simple intraday trading bot for NSE stocks using Python, XGBoost and Telegram alerts.  
This is an **active WIP project** – still being improved with more AI, automation, and future plans for news scraping–based signals to make it more robust.

1. Download / Clone

        git clone https://github.com/<your-username>/<your-repo-name>.git
        cd <your-repo-name>

2. Project Structure
        After download, your folder should look like this:

        prediction/
        ├── config.py
        ├── data/
        ├── logs/
        ├── notebook/
        ├── requirements.txt
        └── src/
          ├── main.py
          ├── model_trainer.py
          ├── data_fetcher.py
          ├── features_intraday.py
          ├── signal_generator.py
          └── alerts.py

   
4. Virtualenv & Requirements
        From inside the prediction/ folder:

        python -m venv .venv

   Windows:

        .venv\Scripts\activate
   Linux/macOS:

            source .venv/bin/activate
        
            pip install -r requirements.txt
    
4. config.py Set Up:

      Feel Free to adjust parameters, profit, loss etc. ; Also can change the which stock to trade too.

      And also if you want to get a notification on TELEGRAM you can set up your Bot and Chat ID to get the notifications.
   
5. Train Models

        cd prediction
        python -m src.model_trainer
   
      This will download intraday data, engineer features and train XGBoost models for your configured stocks.

7. Run the Bot

        cd prediction
        python -m src.main

8. Delete Previous model to train with NEW Data :

       del models\*.pkl
       del data\*.pkl
       del data\*.csv

   Run above bash and run the main.py again.
   
   The bot will:

      Respect NSE market hours
      Fetch latest OHLCV data
      Generate BUY signals using the trained models
      Send Telegram alerts (if configured)
      Log all trades to data/logs/ for later analysis

   Roadmap:

      Better AI models and tuning for more stable performance
      More automation around paper/live mode workflows
      Planned: Add news scraping and sentiment signals to combine price action with news/keywords for smarter entries and exits

Disclaimer
This project is for educational and experimental use only.
Nothing here is financial advice, and profitability is not guaranteed.
Always start with paper trading and use your own risk management.

text
undefined
