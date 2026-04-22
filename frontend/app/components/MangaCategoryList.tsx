import React from "react";
import { View, Text, FlatList, StyleSheet } from "react-native";
import MangaCard from "./MangaCard";
import { Manga } from "@/lib/mangaTypes";

interface MangaCategoryListProps {
  title: string;
  data: Manga[];
}

const MangaCategoryList: React.FC<MangaCategoryListProps> = ({
  title,
  data,
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
