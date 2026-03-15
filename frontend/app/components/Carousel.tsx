import React, { useRef } from "react";
import { View } from "react-native";
import ReanimatedCarousel, {
  ICarouselInstance,
  Pagination,
} from "react-native-reanimated-carousel";
import {
  useSharedValue,
  Extrapolation,
  interpolate,
} from "react-native-reanimated";
import MangaCard from "./MangaCard";

interface Manga {
  id: string;
  attributes: {
    title: { [key: string]: string };
    description: { en?: string };
  };
  relationships: {
    type: string;
    attributes?: { fileName?: string };
  }[];
}

interface MangaCarouselProps {
  data: Manga[];
  onMangaPress: (manga: Manga) => void;
}

const Carousel: React.FC<MangaCarouselProps> = ({ data, onMangaPress }) => {
  const ref = useRef<ICarouselInstance>(null);
  const progress = useSharedValue<number>(0);

  const onPressPagination = (index: number) => {
    ref.current?.scrollTo({
      count: index - progress.value,
      animated: true,
    });
  };

  return (
    <View>
      <ReanimatedCarousel
        ref={ref}
        data={data}
        width={300}
        height={420}
        autoPlay={true}
        autoPlayInterval={3000}
        loop={true}
        mode="parallax"
        modeConfig={{
          parallaxScrollingScale: 0.85,
          parallaxScrollingOffset: 60,
        }}
        style={{
          width: "100%",
          marginTop: 30,
          paddingHorizontal: 50,
        }}
        onProgressChange={(offsetProgress, absoluteProgress) => {
          progress.value = absoluteProgress;
        }}
        renderItem={({ item: manga }) => (
          <MangaCard
            manga={manga}
            width={300}
            height={450}
            onPress={onMangaPress}
          />
        )}
      />

      <Pagination.Custom
        progress={progress}
        data={data}
        size={10}
        dotStyle={{
          borderRadius: 16,
          backgroundColor: "#262626",
        }}
        activeDotStyle={{
          borderRadius: 8,
          width: 25,
          height: 15,
          overflow: "hidden",
          backgroundColor: "#b8b8b8",
        }}
        containerStyle={{
          gap: 8,
          marginBottom: 10,
          alignItems: "center",
          height: 10,
        }}
        horizontal
        onPress={onPressPagination}
        customReanimatedStyle={(progress, index, length) => {
          let val = Math.abs(progress - index);
          if (index === 0 && progress > length - 1) {
            val = Math.abs(progress - length);
          }

          return {
            transform: [
              {
                translateY: interpolate(
                  val,
                  [0, 1],
                  [0, 0],
                  Extrapolation.CLAMP,
                ),
              },
            ],
          };
        }}
      />
    </View>
  );
};

export default Carousel;
