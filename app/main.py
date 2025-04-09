from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.routes import auth, users, patients, caregivers, links, notifications, reminders
from app.database import engine, Base
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def initialize_database():
    """Initialize database tables with error handling"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

# Initialize database before app starts
initialize_database()

app = FastAPI(
    title="NeuralTrace User Management API",
    description="API for managing patients and caregivers in the NeuralTrace system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={
        "name": "Support Team",
        "email": "support@neuraltrace.com"
    },
    license_info={
        "name": "MIT",
    }
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with prefixes
app.include_router(auth.router, prefix="/auth")
# app.include_router(users.router, prefix="/users")
app.include_router(patients.router, prefix="/patients")
app.include_router(caregivers.router, prefix="/caregivers")
app.include_router(links.router, prefix="/links")
app.include_router(notifications.router, prefix="/notifications")
app.include_router(reminders.router, prefix="/reminders")

@app.get("/", tags=["Root"], include_in_schema=False)
def read_root() -> Dict[str, str]:
    """Root endpoint that redirects to documentation"""
    return {
        "message": "Welcome to NeuralTrace User Management API",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get(
    "/health",
    tags=["Health Check"],
    summary="System Health Check",
    description="Public endpoint to verify API health status",
    response_model=Dict[str, str]
)
def health_check() -> Dict[str, str]:
    """Public health check endpoint"""
    return {
        "status": "healthy",
        "details": "All systems operational",
        "database": "connected",
        "version": "1.0.0"
    }

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions with proper formatting"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "success": False,
            "error": exc.__class__.__name__
        },
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception) -> JSONResponse:
    """Handle all other exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "success": False,
            "error": "InternalServerError"
        },
    )