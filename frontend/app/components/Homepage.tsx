import { Text, ScrollView, StyleSheet, View } from "react-native";
import React, { useCallback, useEffect, useState } from "react";
import MangaCarousel from "./MangaCarousel";
import MangaCategoryList from "./MangaCategoryList";
import { Manga } from "@/lib/mangaTypes";
import SearchBar from "./SearchBar";
import { type Href, useRouter } from "expo-router";
import type { MangaSearchListJson } from "@/lib/apiTypes";

import { BACKEND_URL } from "@/config";

export default function Index() {
  const [searchQuery, setSearchQuery] = useState("");
  const [popularManga, setPopularManga] = useState<Manga[]>([]);
  const [recentManga, setRecentManga] = useState<Manga[]>([]);
  const [actionManga, setActionManga] = useState<Manga[]>([]);
  const [romanceManga, setRomanceManga] = useState<Manga[]>([]);
  const [sportsManga, setSportsManga] = useState<Manga[]>([]);
  const [comedyManga, setComedyManga] = useState<Manga[]>([]);
  const [horrorManga, setHorrorManga] = useState<Manga[]>([]);
  const [sciFiManga, setSciFiManga] = useState<Manga[]>([]);
  const [sliceOfLifeManga, setSliceOfLifeManga] = useState<Manga[]>([]);
  const router = useRouter();

  const fetchBackend = useCallback(async (params: string) => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/manga/search?${params}`, {
        headers: {
          "ngrok-skip-browser-warning": "true",
        },
      });
      const json = (await res.json()) as MangaSearchListJson;
      return json.data ?? [];
    } catch {
      return [];
    }
  }, []);

  const loadHomePage = useCallback(async () => {
    const [
      popular,
      recent,
      action,
      romance,
      sports,
      comedy,
      horror,
      sciFi,
      sliceOfLife,
    ] = await Promise.all([
      fetchBackend("limit=10&order_by=followedCount&order_direction=desc"),
      fetchBackend(
        "limit=6&order_by=latestUploadedChapter&order_direction=desc",
      ),
      fetchBackend("limit=6&includedTags=391b0423-d847-456f-aff0-8b0cfc03066b"), // Action
      fetchBackend("limit=6&includedTags=423e2eae-a7a2-4a8b-ac03-a8351462d71d"), // Romance
      fetchBackend("limit=6&includedTags=69964a64-2f90-4d33-beeb-f3ed2875eb4c"), // Sports
      fetchBackend("limit=6&includedTags=4d32cc48-9f00-4cca-9b5a-a839f0764984"), // Comedy
      fetchBackend("limit=6&includedTags=cdad7e68-1419-41dd-bdce-27753074a640"), // Horror
      fetchBackend("limit=6&includedTags=256c8bd9-4904-4360-bf4f-508a76d67183"), // Sci-Fi
      fetchBackend("limit=6&includedTags=e5301a23-ebd9-49dd-a0cb-2add944c7fe9"), // Slice of Life
    ]);

    setPopularManga(popular);
    setRecentManga(recent);
    setActionManga(action);
    setRomanceManga(romance);
    setSportsManga(sports);
    setComedyManga(comedy);
    setHorrorManga(horror);
    setSciFiManga(sciFi);
    setSliceOfLifeManga(sliceOfLife);
  }, [fetchBackend]); 

  useEffect(() => {
    void loadHomePage();
  }, [loadHomePage]);

  const handleSearch = () => {
    const q = searchQuery.trim();
    if (!q) return;
    router.push({
      pathname: "/search",
      params: { query: q },
    } as Href);
  };

  return (
    <ScrollView
      style={{ flex: 1, backgroundColor: "#1f1e32" }}
      contentContainerStyle={{
        alignItems: "stretch",
      }}
    >
      <View 
        style={{backgroundColor: "#33384e", gap: 30}}
      >
        <View style={styles.top}>
          <Text style={styles.h1}>Manglify</Text>
          <SearchBar
            placeholder="Search manga..."
            value={searchQuery}
            onChangeText={setSearchQuery}
            onSubmitEditing={handleSearch}
          />
        </View>
      
      <MangaCarousel data={popularManga} />
      </View>
      <View style={{ paddingHorizontal: 20}}>
        <MangaCategoryList title="Recently Updated" data={recentManga} />
        <MangaCategoryList title="Action" data={actionManga} />
        <MangaCategoryList title="Romance" data={romanceManga} />
        <MangaCategoryList title="Sports" data={sportsManga} />
        <MangaCategoryList title="Comedy" data={comedyManga} />
        <MangaCategoryList title="Horror" data={horrorManga} />
        <MangaCategoryList title="Sci-Fi" data={sciFiManga} />
        <MangaCategoryList title="Slice of Life" data={sliceOfLifeManga} />
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  h1: {
    fontSize: 32,
    fontWeight: "bold",
    flexShrink: 1,
    color: "#fff"
  },
  top: {
    display: "flex",
    flexDirection: "row",
    justifyContent: "space-between",
    width: "100%",
    paddingTop: "2%",
    paddingHorizontal: "5%",
  }
});
