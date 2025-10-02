class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def display_info(self):
        print(f"👤 Name: {self.name}, Age: {self.age}")

class Student(Person):
    def __init__(self, name, age, student_id):
        super().__init__(name, age)
        self.student_id = student_id
        self.courses = []

    def enroll(self, course_name):
        self.courses.append(course_name)
        print(f"✅ {self.name} enrolled in {course_name}")

    def display_info(self):
        super().display_info()
        print(f"🎓 Student ID: {self.student_id}")
        print("📚 Enrolled Courses:", ", ".join(self.courses))

class Teacher(Person):
    def __init__(self, name, age, subject):
        super().__init__(name, age)
        self.subject = subject

    def display_info(self):
        super().display_info()
        print(f"📘 Teaches: {self.subject}")

class Course:
    def __init__(self, course_name, teacher):
        self.course_name = course_name
        self.teacher = teacher
        self.students = []

    def add_student(self, student):
        self.students.append(student)
        student.enroll(self.course_name)

    def display_course(self):
        print(f"📖 Course: {self.course_name}")
        print(f"👨‍🏫 Teacher: {self.teacher.name}")
        print("👥 Students Enrolled:")
        for s in self.students:
            print(f" - {s.name}")


# Create teacher and student
t1 = Teacher("Sir Ali", 35, "Python")
t2 = Teacher("Ms. Fatima", 30, "JavaScript")
s1 = Student("Saadi", 19, "S123")
s2 = Student("Umer", 20, "S456")
s3 = Student("Sara", 18, "S789")

# Create course
python_course = Course("Python Programming", t1)
javascript_course = Course("JavaScript Programming", t2)

# Enroll students
python_course.add_student(s1)
python_course.add_student(s2)
javascript_course.add_student(s3)
javascript_course.add_student(s1)  

# Show details
print("\n--- Teacher ---")
t1.display_info()
t2.display_info()

print("\n--- Students ---")
s1.display_info()
s2.display_info()
s3.display_info()  

print("\n--- Course ---")
python_course.display_course()
javascript_course.display_course()
