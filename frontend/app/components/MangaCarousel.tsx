import React from "react";
import { View, useWindowDimensions, StyleSheet } from "react-native";
import Carousel, { ICarouselInstance } from "react-native-reanimated-carousel";
import MangaCard from "./MangaCard";
import { Manga } from "@/lib/mangaTypes";

interface MangaCarouselProps {
  data: Manga[];
}

const MangaCarousel: React.FC<MangaCarouselProps> = ({ data}) => {
  const { width } = useWindowDimensions();
  const ref = React.useRef<ICarouselInstance>(null);

  return (
    <View style={styles.container}>
      <Carousel
        ref={ref}
        data={data}
        width={300}
        height={450}
        autoPlay
        loop
        style={{
          width: width - 200, // -200 because for page width margin
          justifyContent: "center",
        }}
        mode="parallax"
        modeConfig={{
          parallaxScrollingScale: 1.0,
          parallaxAdjacentItemScale: 0.8,
          parallaxScrollingOffset: 20,
        }}
        renderItem={({ item }) => (
          <View style={styles.cardWrapper}>
            <MangaCard
              manga={item}
              width={300}
              height={400}
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
