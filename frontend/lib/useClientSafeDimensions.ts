import { useEffect, useState } from "react";
import { Platform, useWindowDimensions } from "react-native";

const SSR_FALLBACK_WIDTH = 1200;
const SSR_FALLBACK_HEIGHT = 800;

/**
 * Web static render often sees width 0 (or defaults) while the client has the real
 * viewport, which produces different trees and React #418. Until mount, use a
 * stable fallback so server HTML matches the first client paint.
 */
export function useClientSafeDimensions() {
  const { width, height, scale, fontScale } = useWindowDimensions();
  const [hydrated, setHydrated] = useState(Platform.OS !== "web");

  useEffect(() => {
    setHydrated(true);
  }, []);

  if (Platform.OS === "web" && !hydrated) {
    return {
      width: SSR_FALLBACK_WIDTH,
      height: SSR_FALLBACK_HEIGHT,
      scale,
      fontScale,
    };
  }

  return { width, height, scale, fontScale };
}
