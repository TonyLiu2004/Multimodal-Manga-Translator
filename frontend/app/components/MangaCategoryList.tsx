import React from "react";
import { View, Text, FlatList, StyleSheet } from "react-native";
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

interface MangaCategoryListProps {
  title: string;
  data: Manga[];
  onMangaPress: (manga: Manga) => void;
}

const MangaCategoryList: React.FC<MangaCategoryListProps> = ({
  title,
  data,
  onMangaPress,
}) => {
  return (
    <View style={{ marginTop: 30 }}>
      <Text style={styles.category_header}>{title}</Text>
      <FlatList
        horizontal={true}
        data={data}
        style={styles.category_list}
        renderItem={({ item: manga }) => (
          <MangaCard
            manga={manga}
            width={200}
            height={300}
            onPress={onMangaPress}
          />
        )}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  category_header: {
    fontSize: 21,
    marginBottom: 10,
  },
  category_list: {
    marginLeft: -12,
    marginBottom: 25,
  },
});

export default MangaCategoryList;
