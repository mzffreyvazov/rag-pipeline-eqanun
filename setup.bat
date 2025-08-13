@echo off
REM Azerbaijani Legal RAG Pipeline - Windows Deployment Script

echo 🚀 Starting Azerbaijani Legal RAG Pipeline deployment...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Check if pip is installed
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ pip is not installed. Please install pip.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📚 Installing dependencies...
pip install -r requirements.txt

REM Check if .env file exists
if not exist ".env" (
    echo ⚠️ .env file not found. Please copy .env.example to .env and fill in your credentials.
    copy .env.example .env
    echo 📝 Created .env file from template. Please edit it with your actual credentials.
)

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist "chroma_data" mkdir chroma_data
if not exist "uploads" mkdir uploads
if not exist "logs" mkdir logs

echo ✅ Setup complete!
echo.
echo 📋 Next steps:
echo 1. Edit the .env file with your Google Cloud credentials
echo 2. Run the application with: python app/main.py
echo    Or with uvicorn: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
echo.
echo 🌐 The API will be available at: http://localhost:8000
echo 📖 API documentation at: http://localhost:8000/docs

pause
