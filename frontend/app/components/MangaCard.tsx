import React from "react";
import { View, Image, Pressable } from "react-native";

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

interface MangaCardProps {
  manga: Manga;
  width: number;
  height: number;
  onPress: (manga: Manga) => void;
}

const MangaCard: React.FC<MangaCardProps> = ({
  manga,
  width,
  height,
  onPress,
}) => {
  // const titles = Object.values(manga.attributes.title);
  // const displayTitle = (titles[0] as string) || "Untitled";

  // get cover art url
  const coverArt = manga.relationships.find(
    (rel: any) => rel.type === "cover_art",
  );
  const fileName = coverArt?.attributes?.fileName;
  const coverUrl = fileName
    ? `https://uploads.mangadex.org/covers/${manga.id}/${fileName}.256.jpg`
    : "https://via.placeholder.com/256x360?text=No+Cover";

  return (
    <View
      style={{
        flex: 1,
        alignItems: "center",
        justifyContent: "center",
        marginHorizontal: 12,
      }}
    >
      <Pressable
        style={{
          alignItems: "center",
          justifyContent: "center",
        }}
        onPress={() => onPress(manga)}
      >
        <Image
          source={{ uri: coverUrl }}
          style={{ width: width, height: height, borderRadius: 10 }}
        />
      </Pressable>
    </View>
  );
};

export default MangaCard;
