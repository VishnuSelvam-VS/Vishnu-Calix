import requests
import sys
import json
from datetime import datetime

class VishnuCalixAPITester:
    def __init__(self, base_url="https://calisthenics-ai-1.preview.emergentagent.com/api"):
        self.base_url = base_url
        self.token = None
        self.user_id = None
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name, success, details=""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
        
        result = {
            "test_name": name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {name}")
        if details:
            print(f"   Details: {details}")

    def run_test(self, name, method, endpoint, expected_status, data=None, headers=None):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        test_headers = {'Content-Type': 'application/json'}
        
        if self.token:
            test_headers['Authorization'] = f'Bearer {self.token}'
        
        if headers:
            test_headers.update(headers)

        try:
            if method == 'GET':
                response = requests.get(url, headers=test_headers, timeout=30)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=test_headers, timeout=30)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=test_headers, timeout=30)

            success = response.status_code == expected_status
            details = f"Status: {response.status_code}, Expected: {expected_status}"
            
            if not success:
                try:
                    error_detail = response.json()
                    details += f", Response: {error_detail}"
                except:
                    details += f", Response: {response.text[:200]}"
            
            self.log_test(name, success, details)
            
            if success:
                try:
                    return True, response.json()
                except:
                    return True, {}
            else:
                return False, {}

        except Exception as e:
            self.log_test(name, False, f"Exception: {str(e)}")
            return False, {}

    def test_root_endpoint(self):
        """Test root API endpoint"""
        return self.run_test("Root API Endpoint", "GET", "", 200)

    def test_user_registration(self):
        """Test user registration"""
        timestamp = datetime.now().strftime('%H%M%S')
        user_data = {
            "username": f"testuser_{timestamp}",
            "email": f"test_{timestamp}@example.com",
            "password": "TestPass123!",
            "full_name": "Test User"
        }
        
        success, response = self.run_test(
            "User Registration", 
            "POST", 
            "auth/register", 
            200, 
            data=user_data
        )
        
        if success and 'token' in response:
            self.token = response['token']
            if 'user' in response:
                self.user_id = response['user'].get('id')
            return True
        return False

    def test_user_login(self):
        """Test user login with existing credentials"""
        # Use a test account that should exist
        login_data = {
            "email": "test@example.com",
            "password": "TestPass123!"
        }
        
        success, response = self.run_test(
            "User Login", 
            "POST", 
            "auth/login", 
            200, 
            data=login_data
        )
        
        if success and 'token' in response:
            self.token = response['token']
            if 'user' in response:
                self.user_id = response['user'].get('id')
            return True
        return False

    def test_get_user_profile(self):
        """Test getting user profile"""
        if not self.token:
            self.log_test("Get User Profile", False, "No token available")
            return False
            
        return self.run_test("Get User Profile", "GET", "auth/me", 200)[0]

    def test_fitness_calculation(self):
        """Test fitness calculation"""
        if not self.token:
            self.log_test("Fitness Calculation", False, "No token available")
            return False
            
        fitness_data = {
            "height": 175.0,
            "weight": 70.0,
            "age": 25,
            "activity_level": "moderate"
        }
        
        success, response = self.run_test(
            "Fitness Calculation", 
            "POST", 
            "fitness/calculate", 
            200, 
            data=fitness_data
        )
        
        if success:
            required_fields = ['bmi', 'bmi_category', 'bmr', 'tdee', 'fitness_level', 'recommendations']
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                self.log_test("Fitness Calculation Response Validation", False, f"Missing fields: {missing_fields}")
                return False
            else:
                self.log_test("Fitness Calculation Response Validation", True, "All required fields present")
        
        return success

    def test_progress_tracking(self):
        """Test progress tracking"""
        if not self.token:
            self.log_test("Progress Tracking", False, "No token available")
            return False
            
        # Add progress entry
        progress_data = {
            "weight": 72.5,
            "notes": "Test progress entry"
        }
        
        add_success = self.run_test(
            "Add Progress Entry", 
            "POST", 
            "fitness/progress", 
            200, 
            data=progress_data
        )[0]
        
        # Get progress history
        get_success = self.run_test("Get Progress History", "GET", "fitness/progress", 200)[0]
        
        return add_success and get_success

    def test_ai_chat(self):
        """Test AI chat functionality"""
        if not self.token:
            self.log_test("AI Chat", False, "No token available")
            return False
            
        chat_data = {
            "message": "Hello, can you help me with a basic workout routine?",
            "session_id": None
        }
        
        success, response = self.run_test(
            "AI Chat Message", 
            "POST", 
            "chat/message", 
            200, 
            data=chat_data
        )
        
        if success:
            required_fields = ['response', 'session_id']
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                self.log_test("AI Chat Response Validation", False, f"Missing fields: {missing_fields}")
                return False
            else:
                self.log_test("AI Chat Response Validation", True, "All required fields present")
        
        return success

    def test_workout_generation(self):
        """Test workout plan generation"""
        if not self.token:
            self.log_test("Workout Generation", False, "No token available")
            return False
            
        workout_data = {
            "fitness_level": "beginner",
            "goal": "muscle_gain",
            "days_per_week": 3
        }
        
        success, response = self.run_test(
            "Generate Workout Plan", 
            "POST", 
            "workout/generate", 
            200, 
            data=workout_data
        )
        
        if success:
            required_fields = ['id', 'user_id', 'plan', 'fitness_level', 'goal']
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                self.log_test("Workout Generation Response Validation", False, f"Missing fields: {missing_fields}")
                return False
            else:
                self.log_test("Workout Generation Response Validation", True, "All required fields present")
        
        # Test listing workouts
        list_success = self.run_test("List Workout Plans", "GET", "workout/list", 200)[0]
        
        return success and list_success

    def test_diet_generation(self):
        """Test diet plan generation"""
        if not self.token:
            self.log_test("Diet Generation", False, "No token available")
            return False
            
        diet_data = {
            "goal": "muscle_gain",
            "calories": 2500,
            "preferences": "No dairy"
        }
        
        success, response = self.run_test(
            "Generate Diet Plan", 
            "POST", 
            "diet/generate", 
            200, 
            data=diet_data
        )
        
        if success:
            required_fields = ['id', 'user_id', 'plan', 'goal', 'calories']
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                self.log_test("Diet Generation Response Validation", False, f"Missing fields: {missing_fields}")
                return False
            else:
                self.log_test("Diet Generation Response Validation", True, "All required fields present")
        
        # Test listing diets
        list_success = self.run_test("List Diet Plans", "GET", "diet/list", 200)[0]
        
        return success and list_success

    def test_social_links(self):
        """Test social links update"""
        if not self.token:
            self.log_test("Social Links Update", False, "No token available")
            return False
            
        social_data = {
            "instagram_url": "https://instagram.com/testuser",
            "youtube_url": "https://youtube.com/testuser"
        }
        
        return self.run_test(
            "Update Social Links", 
            "PUT", 
            "social/links", 
            200, 
            data=social_data
        )[0]

    def test_contact_form(self):
        """Test contact form submission"""
        contact_data = {
            "name": "Test User",
            "email": "test@example.com",
            "subject": "Test Message",
            "message": "This is a test message from the API test suite."
        }
        
        return self.run_test(
            "Contact Form Submission", 
            "POST", 
            "contact", 
            200, 
            data=contact_data
        )[0]

    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting Vishnu Calix API Tests...")
        print("=" * 50)
        
        # Test basic connectivity
        self.test_root_endpoint()
        
        # Test authentication flow
        auth_success = self.test_user_registration()
        if not auth_success:
            # Try login if registration fails
            auth_success = self.test_user_login()
        
        if auth_success:
            self.test_get_user_profile()
            
            # Test core features
            self.test_fitness_calculation()
            self.test_progress_tracking()
            
            # Test AI features (may take longer)
            print("\n🤖 Testing AI Features (may take 10-30 seconds)...")
            self.test_ai_chat()
            self.test_workout_generation()
            self.test_diet_generation()
            
            # Test other features
            self.test_social_links()
        else:
            print("⚠️  Authentication failed - skipping authenticated tests")
        
        # Test public endpoints
        self.test_contact_form()
        
        # Print summary
        print("\n" + "=" * 50)
        print(f"📊 Test Summary: {self.tests_passed}/{self.tests_run} tests passed")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All tests passed!")
            return 0
        else:
            print("❌ Some tests failed")
            return 1

def main():
    tester = VishnuCalixAPITester()
    return tester.run_all_tests()

if __name__ == "__main__":
    sys.exit(main())