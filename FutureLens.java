import javax.swing.*;
import java.awt.BorderLayout;
import java.awt.CardLayout;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.event.*;
import java.util.*;
import java.io.*;

// User class must be defined first
class User {
    String username, password, name, email;

    User(String username, String password, String name, String email) {
        this.username = username;
        this.password = password;
        this.name = name;
        this.email = email;
    }
}

public class FutureLens extends JFrame {
    private CardLayout cardLayout;
    private JPanel cardPanel;
    private JTextField nameField, emailField, userField;
    private JPasswordField passField;
    private JProgressBar progressBar;
    private JLabel questionLabel;
    private int score = 0;
    private int questionIndex = 0;
    private String currentUser = "";

    private java.util.List<User> users = new ArrayList<>();
    private java.util.List<String> courses = new ArrayList<>();
    private final String[] questions = {
            "Do you enjoy solving problems?",
            "Are you interested in technology?",
            "Do you like helping others?",
            "Do you enjoy writing?",
            "Are you good with numbers?"
    };

    private final Map<String, java.util.List<String>> courseMap = Map.of(
            "Problem Solving", java.util.List.of("Algorithm Design", "Competitive Programming"),
            "Technology", java.util.List.of("Java Programming", "Web Development"),
            "Helping Others", java.util.List.of("Psychology Basics", "Counseling Skills"),
            "Writing", java.util.List.of("Creative Writing", "Blogging for Beginners"),
            "Numbers", java.util.List.of("Basic Accounting", "Data Analysis with Excel")
    );

    private final File userFile = new File("users.txt");

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new FutureLens().setVisible(true));
    }

    public FutureLens() {
        setTitle("Future Lens - Career Guidance");
        setSize(600, 500);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setLocationRelativeTo(null);

        loadUsers();
        cardLayout = new CardLayout();
        cardPanel = new JPanel(cardLayout);

        cardPanel.add(loginPanel(), "Login");
        cardPanel.add(registerPanel(), "Register");
        cardPanel.add(menuPanel(), "Menu");
        cardPanel.add(assessmentPanel(), "Assessment");
        cardPanel.add(recommendationPanel(), "Recommendation");
        cardPanel.add(coursePanel(), "Courses");

        add(cardPanel);
        cardLayout.show(cardPanel, "Login");
    }

    private JPanel loginPanel() {
        JPanel panel = new JPanel(new GridLayout(6, 1, 10, 10));
        userField = new JTextField();
        passField = new JPasswordField();
        JButton loginBtn = new JButton("Login");
        JButton registerBtn = new JButton("Register");

        panel.setBorder(BorderFactory.createEmptyBorder(50, 100, 50, 100));
        panel.add(new JLabel("Username:"));
        panel.add(userField);
        panel.add(new JLabel("Password:"));
        panel.add(passField);
        panel.add(loginBtn);
        panel.add(registerBtn);

        loginBtn.addActionListener(e -> {
            String user = userField.getText().trim();
            String pass = new String(passField.getPassword());
            if (validateUser(user, pass)) {
                currentUser = user;
                cardLayout.show(cardPanel, "Menu");
            } else {
                JOptionPane.showMessageDialog(this, "Invalid credentials.");
            }
        });

        registerBtn.addActionListener(e -> cardLayout.show(cardPanel, "Register"));
        return panel;
    }

    private JPanel registerPanel() {
        JPanel panel = new JPanel(new GridLayout(10, 1, 10, 10));
        nameField = new JTextField();
        emailField = new JTextField();
        JTextField regUser = new JTextField();
        JPasswordField regPass = new JPasswordField();
        JButton registerBtn = new JButton("Register");
        JButton backBtn = new JButton("Back");

        panel.setBorder(BorderFactory.createEmptyBorder(40, 100, 40, 100));
        panel.add(new JLabel("Name:"));
        panel.add(nameField);
        panel.add(new JLabel("Email:"));
        panel.add(emailField);
        panel.add(new JLabel("Username:"));
        panel.add(regUser);
        panel.add(new JLabel("Password:"));
        panel.add(regPass);
        panel.add(registerBtn);
        panel.add(backBtn);

        registerBtn.addActionListener(e -> {
            String user = regUser.getText().trim();
            String pass = new String(regPass.getPassword());
            String name = nameField.getText().trim();
            String email = emailField.getText().trim();

            if (user.isEmpty() || pass.isEmpty() || name.isEmpty() || email.isEmpty()) {
                JOptionPane.showMessageDialog(this, "Please fill all fields.");
                return;
            }

            for (User u : users) {
                if (u.username.equals(user)) {
                    JOptionPane.showMessageDialog(this, "Username already exists.");
                    return;
                }
            }

            users.add(new User(user, pass, name, email));
            saveUsers();
            JOptionPane.showMessageDialog(this, "Registration successful!");
            cardLayout.show(cardPanel, "Login");
        });

        backBtn.addActionListener(e -> cardLayout.show(cardPanel, "Login"));
        return panel;
    }

    private JPanel menuPanel() {
        JPanel panel = new JPanel(new GridLayout(5, 1, 15, 15));
        JButton startBtn = new JButton("Start Assessment");
        JButton recommendBtn = new JButton("View Recommendation");
        JButton courseBtn = new JButton("View Courses");
        JButton logoutBtn = new JButton("Logout");

        panel.setBorder(BorderFactory.createEmptyBorder(60, 100, 60, 100));
        panel.add(startBtn);
        panel.add(recommendBtn);
        panel.add(courseBtn);
        panel.add(logoutBtn);

        startBtn.addActionListener(e -> {
            score = 0;
            questionIndex = 0;
            courses.clear(); // Clear previous selections
            cardLayout.show(cardPanel, "Assessment");
        });

        recommendBtn.addActionListener(e -> cardLayout.show(cardPanel, "Recommendation"));
        courseBtn.addActionListener(e -> cardLayout.show(cardPanel, "Courses"));
        logoutBtn.addActionListener(e -> cardLayout.show(cardPanel, "Login"));

        return panel;
    }

    private JPanel assessmentPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        questionLabel = new JLabel("", SwingConstants.CENTER);
        JButton yesBtn = new JButton("Yes");
        JButton noBtn = new JButton("No");
        JPanel btnPanel = new JPanel(new FlowLayout());

        btnPanel.add(yesBtn);
        btnPanel.add(noBtn);

        progressBar = new JProgressBar(0, questions.length);
        progressBar.setStringPainted(true);

        panel.add(questionLabel, BorderLayout.NORTH);
        panel.add(btnPanel, BorderLayout.CENTER);
        panel.add(progressBar, BorderLayout.SOUTH);

        updateQuestion();

        yesBtn.addActionListener(e -> {
            score++;
            updateCourseList();
            nextQuestion();
        });

        noBtn.addActionListener(e -> nextQuestion());
        return panel;
    }

    private JPanel recommendationPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        JTextArea area = new JTextArea();
        area.setEditable(false);

        for (User u : users) {
            area.append("Name: " + u.name + "\n");
            area.append("Email: " + u.email + "\n\n");
        }

        panel.add(new JScrollPane(area), BorderLayout.CENTER);
        return panel;
    }

    private JPanel coursePanel() {
        JPanel panel = new JPanel(new BorderLayout());
        JTextArea area = new JTextArea();
        area.setEditable(false);

        for (String c : courses) {
            area.append("- " + c + "\n");
        }

        panel.add(new JScrollPane(area), BorderLayout.CENTER);
        return panel;
    }

    private void updateQuestion() {
        if (questionIndex < questions.length) {
            questionLabel.setText(questions[questionIndex]);
            progressBar.setValue(questionIndex);
            progressBar.setString((questionIndex * 100 / questions.length) + "%");
        }
    }

    private void nextQuestion() {
        questionIndex++;
        if (questionIndex >= questions.length) {
            cardLayout.show(cardPanel, "Recommendation");
        } else {
            updateQuestion();
        }
    }

    private void updateCourseList() {
        switch (questionIndex) {
            case 0 -> courses.addAll(courseMap.get("Problem Solving"));
            case 1 -> courses.addAll(courseMap.get("Technology"));
            case 2 -> courses.addAll(courseMap.get("Helping Others"));
            case 3 -> courses.addAll(courseMap.get("Writing"));
            case 4 -> courses.addAll(courseMap.get("Numbers"));
        }
    }

    private boolean validateUser(String username, String password) {
        for (User u : users) {
            if (u.username.equals(username) && u.password.equals(password)) {
                return true;
            }
        }
        return false;
    }

    private void loadUsers() {
        if (!userFile.exists()) return;
        try (Scanner scanner = new Scanner(userFile)) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                String[] parts = line.split(",");
                if (parts.length == 4) {
                    users.add(new User(parts[0], parts[1], parts[2], parts[3]));
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void saveUsers() {
        try (PrintWriter pw = new PrintWriter(userFile)) {
            for (User u : users) {
                pw.println(u.username + "," + u.password + "," + u.name + "," + u.email);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
