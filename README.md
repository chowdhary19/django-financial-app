
# Django Financial Analysis and Backtesting Application 

- **Author:** Yuvraj Singh Chowdhary  
- **GitHub:** [chowdhary19](https://github.com/chowdhary19)  
- **LinkedIn:** [Yuvraj Singh Chowdhary](https://www.linkedin.com/in/connectyuvraj/) 
- **Portfolio:** [Yuvraj Singh Chowdhary](https://www.linkedin.com/in/connectyuvraj/) 

---

## Overview 

This Django-based financial analysis and backtesting application enables insights into stock performance, backtesting strategies, and stock price predictions. It is designed for **production-grade** usage while also being developer-friendly with support for automated deployment via **GitHub Actions** and **AWS services** for scalability.

The **primary goals** of this project are:
- Enable effective **financial analysis** through a robust **backtesting** framework.
- **Streamline deployment** using Docker and GitHub Actions.
- Provide a **beginner-friendly setup** while maintaining complexity that challenges advanced developers.

The application is live and accessible at:
- **Public URL**: [http://13.127.145.76:8000/stocks/](http://13.127.145.76:8000/stocks/)

---

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Tech Stack](#tech-stack)
4. [Project Setup](#project-setup)
5. [Environment Variables](#environment-variables)
6. [Running the Application Locally](#running-the-application-locally)
7. [Running Migrations and Seeding Data](#running-migrations-and-seeding-data)
8. [Running Tests](#running-tests)
9. [Docker Deployment](#docker-deployment)
10. [AWS RDS Integration](#aws-rds-integration)
11. [CI/CD Pipeline Setup](#ci-cd-pipeline-setup)
12. [API Usage Examples](#api-usage-examples)
13. [Troubleshooting and Debugging](#troubleshooting-and-debugging)
14. [Future Enhancements](#future-enhancements)
15. [Author](#author)

---

## Features

1. **Stock Data Collection and Storage**: Fetch data from **Alpha Vantage API** and store it in a **PostgreSQL** database.
2. **Backtesting Strategies**: Implement multiple backtesting strategies to evaluate stock performance.
3. **Stock Price Prediction**: Utilize **pre-trained machine learning models** (XGBoost, LightGBM) for stock price prediction.
4. **Automated Deployment**: Fully integrated **CI/CD pipeline** with GitHub Actions.
5. **Task Scheduling**: Use **Celery** and **Celery Beat** for scheduling and executing asynchronous tasks.
6. **Dockerized Services**: Containerized application for easy deployment using **Docker**.
7. **Scalable Architecture**: Use of **AWS RDS** for database and **Redis** for caching and queue management.

---

## Prerequisites

- **Python 3.9+** installed.
- **Docker** and **Docker Compose** installed.
- **AWS Account** with permissions for EC2, RDS, and S3.
- **GitHub Account** for repository and CI/CD integration.
- **Alpha Vantage API Key** for stock data collection.

## Tech Stack

- **Backend Framework**: Django 4.1
- **Database**: PostgreSQL (using AWS RDS)
- **Message Broker**: Redis
- **Asynchronous Tasks**: Celery
- **Machine Learning Libraries**: XGBoost, LightGBM, Scikit-Learn
- **Containerization**: Docker
- **Deployment**: AWS EC2, GitHub Actions
- **API Documentation**: DRF-YASG for interactive API documentation

---

## Project Setup

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone [YOUR GITHUB REPOSITORY URL]
cd django-financial-app
```

### 2. Install Dependencies

Create a **virtual environment** and install the dependencies from `requirements.txt`:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Environment Variables

Refer to the `.env.example` file in the root directory of the project for the required environment variables.

Create a `.env` file in the root directory:

```env
# Django settings
DJANGO_SECRET_KEY=your-secret-key-here
DEBUG=True/False

# Database settings
DATABASE_ENGINE=your-db-engine
DATABASE_NAME=your-db-name
DATABASE_USER=your-db-user-name
DATABASE_PASSWORD=your-db-password
DATABASE_HOST=set-host
DATABASE_PORT=set-port

# Celery settings
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Stock API Keys
ALPHAVANTAGE_API_KEYS=your-key-here

# Testing Environment (Set to True when running tests)
TESTING=True/False 

# Machine learning model path for prediction
MODEL_FILE_PATH=path_to_model     
```

Copy the `.env.example` to `.env` and fill in the appropriate values:

```bash
cp .env.example .env
```

---

## Running the Application Locally

To run the Django server locally, first apply the migrations:

```bash
python manage.py migrate
```

Next, run the development server:

```bash
python manage.py runserver
```

Navigate to `http://127.0.0.1:8000/` to access the application.

## Running Migrations and Seeding Data

To run the database migrations, execute:

```bash
python manage.py migrate
```

To create sample data in the database, use the management command:

```bash
python manage.py loaddata <fixture_file>
```
Replace `<fixture_file>` with the appropriate fixture for data population, if available.

## Running Tests

To ensure that all parts of the application are functioning as intended, you can run the tests:

```bash
sed -i "s/TESTING=False/TESTING=True/g" .env
python manage.py test stocks.tests
sed -i "s/TESTING=True/TESTING=False/g" .env
```

## Docker Deployment

The application is containerized using **Docker**. To build and run the Docker containers:

```bash
docker-compose up -d --build
```

## AWS RDS Integration

Ensure your RDS is properly set up, and the **security group rules** allow inbound traffic to port `5432` from the **EC2 instance's IP**.

## CI/CD Pipeline Setup

The project uses **GitHub Actions** for CI/CD. Add the `.github/workflows/deploy.yml` file to your repository, containing the necessary setup for automated deployment.

---

## API Usage Examples

### 1. Fetch Stock Data

**Endpoint**: `http://13.127.145.76:8000/stocks/fetch/?symbol=AAPL`

**Method**: `GET`

**Description**:
- Fetches stock data for the provided symbol and stores it in the database.

### 2. Backtest Strategy

**Endpoint**: `http://13.127.145.76:8000/stocks/backtest/?initial_investment=10000&symbol=JPM&buy_window=10&sell_window=50`

**Method**: `GET`

**Description**:
- Runs a backtesting strategy for the provided symbol with the specified initial investment and buy/sell windows.

### 3. Predict Stock Prices

**Endpoint**: `http://13.127.145.76:8000/stocks/predict/?symbol=AAPL`

**Method**: `GET`

**Description**:
- Predicts future stock prices for the provided symbol.

### 4. Generate Report

**Endpoint**: 
- JSON: `http://13.127.145.76:8000/stocks/report/?symbol=AAPL&format=json`
- PDF: `http://13.127.145.76:8000/stocks/report/?symbol=AAPL&format=pdf`

**Method**: `GET`

**Description**:
- Generates a report in the requested format (JSON or PDF) summarizing stock performance.

### 5. Export Data

**Endpoint**: `http://13.127.145.76:8000/stocks/fetch/?symbol=AAPL`

**Method**: `GET`

**Description**:
- Exports the stock data for the requested symbol.

---

## Troubleshooting and Debugging

- Ensure AWS RDS security group allows inbound connections on port **5432**.
- Confirm **environment variables** are set correctly in `.env`.
- **Docker** services can be checked using `docker-compose ps` to confirm that everything is running smoothly.

## Future Enhancements

- Implement **Kubernetes** for container orchestration.
- Integrate **ELK stack** for real-time monitoring.
- Enhance prediction accuracy using **Deep Learning** models.

---

This **README** is designed to clearly explain every aspect of the application setup and deployment, making it straightforward for developers at any level to get started while challenging enough for seasoned engineers.




