import React from 'react';
import { View, Text, Image, Pressable, useWindowDimensions } from 'react-native';

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

interface SearchResultsProps {
  mangaList: Manga[];
  onMangaPress: (manga: Manga) => void;
}

const SearchResults: React.FC<SearchResultsProps> = ({ mangaList, onMangaPress }) => {
  const { width } = useWindowDimensions();
  const isDesktop = width > 600;

  if (mangaList.length === 0) {
    return null;
  }

  return (
    <View style={{ marginTop: 20, alignItems: "center" }}>
      <Text>{mangaList.length} manga found!</Text>
      <View style={{ marginTop: 20, alignItems: "center" }}>
        {mangaList.map((manga) => {
          const titles = Object.values(manga.attributes.title);
          const displayTitle = (titles[0] as string) || "Untitled";

          // get cover art url
          const coverArt = manga.relationships.find(
            (rel: any) => rel.type === "cover_art",
          );
          const fileName = coverArt?.attributes?.fileName;
          const coverUrl = fileName
            ? `https://uploads.mangadex.org/covers/${manga.id}/${fileName}.256.jpg`
            : "https://via.placeholder.com/256x360?text=No+Cover";

          return (
            <Pressable
              key={manga.id}
              style={{
                marginBottom: 10,
                justifyContent: "flex-start",
                flexDirection: isDesktop ? "row" : "column",
                width: "50%",
              }}
              onPress={() => onMangaPress(manga)}
            >
              <Image
                source={{ uri: coverUrl }}
                style={{ width: 256, height: 360 }}
              />
              <View style={{ flex: 1, marginLeft: 30 }}>
                <Text style={{ fontWeight: "bold" }}>{displayTitle}</Text>
                <Text style={{ color: "gray" }}>ID: {manga.id}</Text>
                <Text>
                  {manga.attributes.description.en || "No description available."}
                </Text>
              </View>
            </Pressable>
          );
        })}
      </View>
    </View>
  );
};

export default SearchResults;