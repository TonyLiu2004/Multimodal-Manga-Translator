import Ionicons from "@expo/vector-icons/Ionicons";
import { useFonts } from "expo-font";
import { Stack } from "expo-router";
import * as SplashScreen from "expo-splash-screen";
import { useEffect } from "react";
import { AuthProvider } from "@/context/AuthContext";

SplashScreen.preventAutoHideAsync();

export default function RootLayout() {
  // Static / react-server bundle exposes a stub `useFonts` that returns [] — don't treat that as “still loading”.
  const fontHook = useFonts(Ionicons.font);
  const raw = fontHook as unknown as readonly unknown[];
  const serverStub = raw.length === 0;
  const fontsLoaded = raw[0] as boolean | undefined;
  const fontError = raw[1] as Error | null | undefined;

  useEffect(() => {
    if (serverStub || fontsLoaded || fontError != null) {
      SplashScreen.hideAsync().catch(() => {});
    }
  }, [serverStub, fontsLoaded, fontError]);

  return (
    <AuthProvider>
      <Stack
        screenOptions={{
          headerShown: false,
        }}
      />
    </AuthProvider>
  );
}
