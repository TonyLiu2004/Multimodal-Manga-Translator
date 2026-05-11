import React from "react";
import { View, Text, FlatList, StyleSheet, Platform } from "react-native";
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
  const categoryScrollStyle =
    Platform.OS === "web"
      ? ({ scrollbarWidth: "thin", scrollbarColor: "#9daabb transparent" } as any)
      : undefined;

  return (
    <View style={{ marginTop: 30, width: "100%" }}>
      <Text style={styles.category_header}>{title}</Text>
      <FlatList
        horizontal={true}
        data={data}
        style={[styles.category_list, categoryScrollStyle]}
        contentContainerStyle={styles.category_list_content}
        showsHorizontalScrollIndicator={true}
        renderItem={({ item: manga }) => (
          <MangaCard
            manga={manga}
            width={Platform.OS === "android" ? 150 : 200}
            height={Platform.OS === "android" ? 230 : 300}
          />
        )}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  category_header: {
    fontSize: 18,
    marginBottom: 10,
    color: "#fff",
  },
  category_list: {
    marginLeft: -12,
    marginBottom: 25,
  },
  category_list_content: {
    paddingLeft: 12,
    paddingRight: 12,
  },
});

export default MangaCategoryList;
