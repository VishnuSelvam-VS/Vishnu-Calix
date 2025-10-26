from fastapi import FastAPI, APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
import jwt
from passlib.context import CryptContext
from emergentintegrations.llm.chat import LlmChat, UserMessage

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# JWT Configuration
SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'vishnu-calix-secret-key-2024')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Create the main app
app = FastAPI()
api_router = APIRouter(prefix="/api")

# Models
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    email: str
    full_name: Optional[str] = None
    height: Optional[float] = None  # in cm
    weight: Optional[float] = None  # in kg
    age: Optional[int] = None
    fitness_level: Optional[str] = None
    goals: Optional[List[str]] = []
    instagram_url: Optional[str] = None
    youtube_url: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class FitnessData(BaseModel):
    height: float
    weight: float
    age: int
    activity_level: str  # sedentary, light, moderate, active, very_active

class FitnessResult(BaseModel):
    bmi: float
    bmi_category: str
    bmr: float  # Basal Metabolic Rate
    tdee: float  # Total Daily Energy Expenditure
    fitness_level: str
    recommendations: List[str]

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

class WorkoutPlanRequest(BaseModel):
    fitness_level: str  # beginner, intermediate, advanced
    goal: str  # muscle_gain, fat_loss, maintenance, endurance
    days_per_week: int

class WorkoutPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    plan: str
    fitness_level: str
    goal: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class DietPlanRequest(BaseModel):
    goal: str  # muscle_gain, fat_loss, maintenance
    calories: int
    preferences: Optional[str] = None

class DietPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    plan: str
    goal: str
    calories: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ProgressEntryRequest(BaseModel):
    weight: float
    notes: Optional[str] = None

class ProgressEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    weight: float
    notes: Optional[str] = None
    date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class SocialLinks(BaseModel):
    instagram_url: Optional[str] = None
    youtube_url: Optional[str] = None

class ContactMessage(BaseModel):
    name: str
    email: EmailStr
    subject: str
    message: str

# Helper Functions
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user = await db.users.find_one({"id": user_id}, {"_id": 0})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return User(**user)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

def calculate_bmi(weight: float, height: float) -> float:
    """Calculate BMI (kg/m^2)"""
    height_m = height / 100
    return round(weight / (height_m ** 2), 2)

def get_bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def calculate_bmr(weight: float, height: float, age: int, gender: str = "male") -> float:
    """Calculate Basal Metabolic Rate using Mifflin-St Jeor Equation"""
    if gender.lower() == "male":
        return round(10 * weight + 6.25 * height - 5 * age + 5, 2)
    else:
        return round(10 * weight + 6.25 * height - 5 * age - 161, 2)

def calculate_tdee(bmr: float, activity_level: str) -> float:
    """Calculate Total Daily Energy Expenditure"""
    activity_multipliers = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9
    }
    multiplier = activity_multipliers.get(activity_level.lower(), 1.2)
    return round(bmr * multiplier, 2)

# Routes
@api_router.get("/")
async def root():
    return {"message": "Vishnu Calix API - Train Smarter, Live Healthier"}

# Auth Routes
@api_router.post("/auth/register")
async def register(user_data: UserCreate):
    # Check if user exists
    existing_user = await db.users.find_one({"email": user_data.email}, {"_id": 0})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    user = User(
        username=user_data.username,
        email=user_data.email,
        full_name=user_data.full_name
    )
    
    doc = user.model_dump()
    doc['password'] = hash_password(user_data.password)
    doc['created_at'] = doc['created_at'].isoformat()
    
    await db.users.insert_one(doc)
    
    # Create token
    token = create_access_token({"sub": user.id, "email": user.email})
    
    return {"token": token, "user": user}

@api_router.post("/auth/login")
async def login(credentials: UserLogin):
    user = await db.users.find_one({"email": credentials.email}, {"_id": 0})
    if not user or not verify_password(credentials.password, user['password']):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Convert ISO string back to datetime
    if isinstance(user['created_at'], str):
        user['created_at'] = datetime.fromisoformat(user['created_at'])
    
    user_obj = User(**{k: v for k, v in user.items() if k != 'password'})
    token = create_access_token({"sub": user_obj.id, "email": user_obj.email})
    
    return {"token": token, "user": user_obj}

@api_router.get("/auth/me")
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user

# Fitness Routes
@api_router.post("/fitness/calculate", response_model=FitnessResult)
async def calculate_fitness(data: FitnessData, current_user: User = Depends(get_current_user)):
    bmi = calculate_bmi(data.weight, data.height)
    bmi_category = get_bmi_category(bmi)
    bmr = calculate_bmr(data.weight, data.height, data.age)
    tdee = calculate_tdee(bmr, data.activity_level)
    
    # Determine fitness level
    if data.activity_level in ["sedentary", "light"]:
        fitness_level = "Beginner"
    elif data.activity_level == "moderate":
        fitness_level = "Intermediate"
    else:
        fitness_level = "Advanced"
    
    # Generate recommendations
    recommendations = []
    if bmi < 18.5:
        recommendations.append("Focus on strength training and increase caloric intake")
    elif bmi >= 25:
        recommendations.append("Incorporate cardio and maintain caloric deficit")
    else:
        recommendations.append("Maintain balanced diet and regular exercise")
    
    recommendations.append(f"Daily calorie target: {int(tdee)} calories")
    recommendations.append("Stay hydrated - drink at least 3L water daily")
    recommendations.append("Get 7-9 hours of quality sleep")
    
    # Update user profile
    await db.users.update_one(
        {"id": current_user.id},
        {"$set": {
            "height": data.height,
            "weight": data.weight,
            "age": data.age,
            "fitness_level": fitness_level
        }}
    )
    
    return FitnessResult(
        bmi=bmi,
        bmi_category=bmi_category,
        bmr=bmr,
        tdee=tdee,
        fitness_level=fitness_level,
        recommendations=recommendations
    )

@api_router.post("/fitness/progress")
async def add_progress(entry_request: ProgressEntryRequest, current_user: User = Depends(get_current_user)):
    entry = ProgressEntry(
        user_id=current_user.id,
        weight=entry_request.weight,
        notes=entry_request.notes
    )
    doc = entry.model_dump()
    doc['date'] = doc['date'].isoformat()
    await db.progress.insert_one(doc)
    return {"message": "Progress recorded", "entry": entry}

@api_router.get("/fitness/progress")
async def get_progress(current_user: User = Depends(get_current_user)):
    progress_list = await db.progress.find({"user_id": current_user.id}, {"_id": 0}).sort("date", -1).to_list(100)
    for p in progress_list:
        if isinstance(p['date'], str):
            p['date'] = datetime.fromisoformat(p['date'])
    return progress_list

# AI Chat Routes
@api_router.post("/chat/message", response_model=ChatResponse)
async def chat_with_coach(chat_msg: ChatMessage, current_user: User = Depends(get_current_user)):
    try:
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        session_id = chat_msg.session_id or f"user-{current_user.id}-{uuid.uuid4()}"
        
        system_message = f"""You are Vishnu Coach, an expert AI fitness coach specializing in calisthenics and bodyweight training. 
        You help users with:
        - Workout form and technique
        - Personalized exercise recommendations
        - Diet and nutrition advice
        - Motivation and progress tracking
        - Home workout optimization
        
        User Profile:
        - Name: {current_user.username}
        - Fitness Level: {current_user.fitness_level or 'Not set'}
        - Height: {current_user.height or 'Not set'} cm
        - Weight: {current_user.weight or 'Not set'} kg
        - Age: {current_user.age or 'Not set'}
        
        Be encouraging, specific, and actionable in your advice."""
        
        chat = LlmChat(
            api_key=api_key,
            session_id=session_id,
            system_message=system_message
        ).with_model("openai", "gpt-5")
        
        user_message = UserMessage(text=chat_msg.message)
        response = await chat.send_message(user_message)
        
        # Store chat history
        chat_doc = {
            "id": str(uuid.uuid4()),
            "user_id": current_user.id,
            "session_id": session_id,
            "user_message": chat_msg.message,
            "ai_response": response,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        await db.chat_history.insert_one(chat_doc)
        
        return ChatResponse(response=response, session_id=session_id)
    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat service error: {str(e)}")

# Workout Routes
@api_router.post("/workout/generate")
async def generate_workout(request: WorkoutPlanRequest, current_user: User = Depends(get_current_user)):
    try:
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        
        prompt = f"""Create a detailed {request.days_per_week}-day per week calisthenics workout plan for a {request.fitness_level} level athlete.
        Goal: {request.goal}
        
        Include:
        - Specific exercises with sets and reps
        - Rest periods
        - Progressive overload suggestions
        - Form tips
        
        Format the plan clearly by day."""
        
        chat = LlmChat(
            api_key=api_key,
            session_id=f"workout-gen-{uuid.uuid4()}",
            system_message="You are an expert calisthenics coach creating workout plans."
        ).with_model("openai", "gpt-5")
        
        user_message = UserMessage(text=prompt)
        plan = await chat.send_message(user_message)
        
        workout = WorkoutPlan(
            user_id=current_user.id,
            plan=plan,
            fitness_level=request.fitness_level,
            goal=request.goal
        )
        
        doc = workout.model_dump()
        doc['created_at'] = doc['created_at'].isoformat()
        await db.workout_plans.insert_one(doc)
        
        return workout
    except Exception as e:
        logging.error(f"Workout generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Workout generation error: {str(e)}")

@api_router.get("/workout/list")
async def list_workouts(current_user: User = Depends(get_current_user)):
    workouts = await db.workout_plans.find({"user_id": current_user.id}, {"_id": 0}).sort("created_at", -1).to_list(50)
    for w in workouts:
        if isinstance(w['created_at'], str):
            w['created_at'] = datetime.fromisoformat(w['created_at'])
    return workouts

# Diet Routes
@api_router.post("/diet/generate")
async def generate_diet(request: DietPlanRequest, current_user: User = Depends(get_current_user)):
    try:
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        
        prompt = f"""Create a detailed daily meal plan for:
        Goal: {request.goal}
        Daily Calories: {request.calories}
        Preferences: {request.preferences or 'None'}
        
        Include:
        - Breakfast, Lunch, Dinner, and 2 Snacks
        - Macronutrient breakdown
        - Portion sizes
        - Meal timing suggestions
        
        Focus on whole foods and sustainable eating."""
        
        chat = LlmChat(
            api_key=api_key,
            session_id=f"diet-gen-{uuid.uuid4()}",
            system_message="You are an expert nutritionist specializing in fitness and sports nutrition."
        ).with_model("openai", "gpt-5")
        
        user_message = UserMessage(text=prompt)
        plan = await chat.send_message(user_message)
        
        diet = DietPlan(
            user_id=current_user.id,
            plan=plan,
            goal=request.goal,
            calories=request.calories
        )
        
        doc = diet.model_dump()
        doc['created_at'] = doc['created_at'].isoformat()
        await db.diet_plans.insert_one(doc)
        
        return diet
    except Exception as e:
        logging.error(f"Diet generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Diet generation error: {str(e)}")

@api_router.get("/diet/list")
async def list_diets(current_user: User = Depends(get_current_user)):
    diets = await db.diet_plans.find({"user_id": current_user.id}, {"_id": 0}).sort("created_at", -1).to_list(50)
    for d in diets:
        if isinstance(d['created_at'], str):
            d['created_at'] = datetime.fromisoformat(d['created_at'])
    return diets

# Social Routes
@api_router.put("/social/links")
async def update_social_links(links: SocialLinks, current_user: User = Depends(get_current_user)):
    await db.users.update_one(
        {"id": current_user.id},
        {"$set": {
            "instagram_url": links.instagram_url,
            "youtube_url": links.youtube_url
        }}
    )
    return {"message": "Social links updated"}

# Contact Route
@api_router.post("/contact")
async def send_contact_message(message: ContactMessage):
    doc = {
        "id": str(uuid.uuid4()),
        "name": message.name,
        "email": message.email,
        "subject": message.subject,
        "message": message.message,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    await db.contact_messages.insert_one(doc)
    return {"message": "Message sent successfully"}

# Include router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()