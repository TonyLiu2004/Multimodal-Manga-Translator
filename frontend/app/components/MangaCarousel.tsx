import React from "react";
import { View, useWindowDimensions, StyleSheet } from "react-native";
import Carousel, { ICarouselInstance } from "react-native-reanimated-carousel";
import MangaCard from "./MangaCard";
import { Manga } from "../types/types";

interface MangaCarouselProps {
  data: Manga[];
}

const MangaCarousel: React.FC<MangaCarouselProps> = ({ data}) => {
  const { width } = useWindowDimensions();
  const ref = React.useRef<ICarouselInstance>(null);

  // Responsive card dimensions
  const isMobile = width < 600;
  const isTablet = width >= 600 && width < 1024;
  
  // Card dimensions
  const cardWidth = isMobile ? Math.min(width * 0.7, 280) : 300;
  const cardHeight = isMobile ? cardWidth * 1.4 : 450;
  
  // Carousel container width - constrain to reasonable max
  let carouselWidth;
  if (isMobile) {
    carouselWidth = width - 40;
  } else if (isTablet) {
    carouselWidth = Math.min(width - 100, 700);
  } else {
    carouselWidth = Math.min(width - 200, 900);
  }

  if (!data || data.length === 0) {
    return null;
  }

  return (
    <View style={styles.container}>
      <Carousel
        ref={ref}
        data={data}
        width={cardWidth}
        height={cardHeight}
        autoPlay
        loop
        style={{
          width: carouselWidth,
          justifyContent: "center",
        }}
        mode="parallax"
        modeConfig={{
          parallaxScrollingScale: 1.0,
          parallaxAdjacentItemScale: isMobile ? 0.75 : 0.8,
          parallaxScrollingOffset: isMobile ? 15 : 20,
        }}
        renderItem={({ item }) => (
          <View style={styles.cardWrapper}>
            <MangaCard
              manga={item}
              width={cardWidth}
              height={cardHeight - 50}
            />
          </View>
        )}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    alignItems: "center",
    justifyContent: "center",
    marginVertical: 20,
  },
  cardWrapper: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
});

export default MangaCarousel;