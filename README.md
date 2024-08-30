# aws-ils-be Code Repository
This is a backend Python Flask REST API application for a system that enables Interactive Learning with Safeguards (ILS).

Paper Publication: https://doi.org/10.1016/j.mlwa.2024.100583

## Pre-requisites
- **Python Environment**

  Create a new `Python 3.10` environment via one of the following ways:
    
  - [Python Virtual Environment (i.e. venv)](https://towardsdatascience.com/virtual-environments-104c62d48c54)
  - [Conda Environment](https://towardsdatascience.com/a-guide-to-conda-environments-bc6180fc533)

- **Install Python Packages**

  After setting up the Python environment and activating it, navigate to the main folder and run the following command to install the required packages.
  ```bash
  pip install -r requirements.txt
  ```

## Running the Codes Locally
  1) Run the following command to get the application running.
     ```bash
     python app.py
     ``` 
  2) View the `Swagger` documentation of 3 REST APIs at `localhost:5000`. Quick testing of each of the APIs can also be done on the Swagger UI.

  ## Running the Codes via Docker

If you have [Docker](https://docs.docker.com/get-docker/) available in your system, the Flask application can also be run as a Docker container. For convenience, scripts are provided to facilitate loading the Docker images and running the Docker containers. You can navigate to the `/deployment` folder and execute the commands below.

- For building Docker images from the code base
```bash
./dockerise.bat
```
- For running Docker containers
```bash
./run-be.bat
```

To stop the Docker container, navigate to the `/deployment` folder and execute the command below:
```bash
./stop-be.bat
```