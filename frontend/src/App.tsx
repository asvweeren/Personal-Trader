import { Component, type ReactNode } from "react";
import { Dashboard } from "./components/Dashboard";
import { LoginPage } from "./components/LoginPage";
import { AuthProvider, useAuth } from "./contexts/AuthContext";

class ErrorBoundary extends Component<
  { children: ReactNode },
  { hasError: boolean; error: string }
> {
  state = { hasError: false, error: "" };

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error: error.message };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gray-950 flex items-center justify-center">
          <div className="text-center">
            <div className="text-red-400 text-lg mb-2">Something went wrong</div>
            <div className="text-gray-500 text-sm mb-4">{this.state.error}</div>
            <button
              onClick={() => {
                this.setState({ hasError: false, error: "" });
                window.location.reload();
              }}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm"
            >
              Reload
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

function AppContent() {
  const { isAuthenticated } = useAuth();
  return isAuthenticated ? <Dashboard /> : <LoginPage />;
}

export default function App() {
  return (
    <ErrorBoundary>
      <AuthProvider>
        <AppContent />
      </AuthProvider>
    </ErrorBoundary>
  );
}
