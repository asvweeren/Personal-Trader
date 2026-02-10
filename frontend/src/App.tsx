import { Dashboard } from "./components/Dashboard";
import { LoginPage } from "./components/LoginPage";
import { AuthProvider, useAuth } from "./contexts/AuthContext";

function AppContent() {
  const { isAuthenticated } = useAuth();
  return isAuthenticated ? <Dashboard /> : <LoginPage />;
}

export default function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}
