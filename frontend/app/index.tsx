import { useWindowDimensions, Text, ScrollView, View, StyleSheet, ActivityIndicator, TextInput, Image, Modal, TouchableOpacity, Button, Linking } from "react-native";
import React, { useEffect, useState } from 'react';

const BASE_URL = "https://api.mangadex.org";

export default function Index() {
  const [searchQuery, setSearchQuery] = useState('');
  const [mangaList, setMangaList] = useState<any[]>([]);
  const [selectedManga, setSelectedManga] = useState<any | null>(null);
  const [chapters, setChapters] = useState<any[]>([]);
  const [loadingChapters, setLoadingChapters] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  // const [chapLang, setChapLang] = useState('en');
  const [totalChapters, setTotalChapters] = useState<number | null>(null);

  const { width } = useWindowDimensions();
  const isDesktop = width > 600;


  const handleSearch = async () => {
    console.log("Searching for:", searchQuery);
    if (!searchQuery) return;
    try {
      const response = await fetch(`https://api.mangadex.org/manga?title=${searchQuery}&limit=10&includes[]=cover_art`);
      const json = await response.json();
      setMangaList(json.data || []);
      console.log("Search results:", json.data);
    } catch (error) {
      console.error("Search error:", error);
    }
  };

  const fetchChapters = async (mangaId: string, lang = '') => {
    setLoadingChapters(true);
    try {
      const response = await fetch(`${BASE_URL}/manga/${mangaId}/feed?order[chapter]=asc`);
      // check for HTTP errors
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const json = await response.json();
      setChapters(json.data || []);
      if (typeof json.total === 'number') setTotalChapters(json.total);
      else setTotalChapters(null);
    } catch (error) {
      console.error('Failed to fetch chapters', error);
      setChapters([]);
      setTotalChapters(null);
    } finally {
      setLoadingChapters(false);
    }
  };

  const handleOpenManga = async (manga: any) => {
    setSelectedManga(manga);
    setModalVisible(true);
    await fetchChapters(manga.id, '');
  };

  return (
    <ScrollView contentContainerStyle={{ flexGrow: 1, alignItems: 'center', padding: 20, justifyContent: 'center' }}>
      <Text style={styles.h1}>Testing</Text>
      <TextInput
        style={styles.input}
        onChangeText={(query) => setSearchQuery(query)}
        onSubmitEditing={handleSearch}
        value={searchQuery}
        placeholder="Search manga title"
      />
      
      {mangaList.length > 0 && (
        <View style={{ marginTop: 20, alignItems: 'center' }}>
          <Text>{mangaList.length} manga found!</Text>
          <View style={{ marginTop: 20, alignItems: 'center' }}>
            {mangaList.map((manga) => {
              const titles = Object.values(manga.attributes.title);
              const displayTitle = titles[0] as string || "Untitled";

              // get cover art url
              const coverArt = manga.relationships.find((rel: any) => rel.type === 'cover_art');
              const fileName = coverArt?.attributes?.fileName;
              const coverUrl = fileName 
                ? `https://uploads.mangadex.org/covers/${manga.id}/${fileName}.256.jpg`
                : 'https://via.placeholder.com/256x360?text=No+Cover';
              return (
                <TouchableOpacity key={manga.id} onPress={() => handleOpenManga(manga)} activeOpacity={0.8} style={
                  {
                    marginBottom: 10, 
                    justifyContent: 'flex-start',
                    flexDirection: isDesktop ? 'row' : 'column',
                    width: '50%',
                  }}>
                    <Image 
                      source={{ uri: coverUrl }} 
                      style={{ width: 256, height: 360 }}
                    />
                    <View style={{ flex: 1, marginLeft: 30 }}>
                      <Text style={{ fontWeight: 'bold' }}>{displayTitle}</Text>
                      <Text style={{ color: 'gray' }}>ID: {manga.id}</Text>
                      <Text>{manga.attributes.description?.en || "No description available."}</Text>
                    </View>
                </TouchableOpacity>
              );
            })}
          </View>
        </View>
      )}

      <Modal visible={modalVisible} animationType="slide" onRequestClose={() => setModalVisible(false)}>
        <View style={modalStyles.container}>
          <View style={modalStyles.header}>
            <Text style={modalStyles.title}>{selectedManga ? Object.values(selectedManga.attributes.title)[0] : 'Chapters'}</Text>
            <Button title="Close" onPress={() => setModalVisible(false)} />
          </View>
          {loadingChapters ? (
            <ActivityIndicator size="large" />
          ) : (
            <View style={{ flex: 1 }}>
              <View style={modalStyles.filterRow}>
                <Text style={{ marginRight: 8 }}>Language:</Text>
                {/* // TODO: language filter buttons, currently just fetches all available chapters regardless of language*/}
                {/* <Button title={chapLang === 'en' ? 'EN ✓' : 'EN'} onPress={async () => { setChapLang('en'); await fetchChapters(selectedManga.id, 'en'); }} /> */}
                {/* <Button title={chapLang === '' ? 'All ✓' : 'All'} onPress={async () => { setChapLang(''); await fetchChapters(selectedManga.id, ''); }} /> */}
                {<Button title='All' onPress={async () => {await fetchChapters(selectedManga.id, ''); }} />}
              </View>
              <ScrollView contentContainerStyle={{ padding: 16 }}>
                {chapters.length === 0 && (
                  <Text>No chapters found.</Text>
                )}
                {chapters.map((ch: any) => {
                  const attrs = ch.attributes || {};
                  const chapNum = attrs.chapter || 'Special';
                  const chapTitle = attrs.title || '';
                  return (
                    // currently opens chapter on mangadex, but ideally would open in built-in reader wired up to our translation features
                    <TouchableOpacity key={ch.id} onPress={() => {
                      const url = `https://mangadex.org/chapter/${ch.id}`;
                      Linking.openURL(url).catch(err => console.error('Failed to open URL', err));
                    }}>
                      <View style={modalStyles.chapterRow}>
                        <Text style={modalStyles.chapterText}>{chapNum} — {chapTitle}</Text>
                      </View>
                    </TouchableOpacity>
                  );
                })}
              </ScrollView>
            </View>
          )}
        </View>
      </Modal>

    </ScrollView>
  );
}

const styles = StyleSheet.create({
  h1: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#333',
    marginVertical: 10,
  },
  input: {
    height: 50,
    width: '50%',
    borderWidth: 1,
    borderColor: '#E0E0E0', // Light border color
    borderRadius: 10,
    paddingHorizontal: 15,
    fontSize: 16,
    color: '#333',          // Color of the text YOU type
    backgroundColor: '#FAFAFA',
  },
});

const modalStyles = StyleSheet.create({
  container: {
    flex: 1,
    marginHorizontal: 20,
    paddingTop: 40,
    backgroundColor: '#fff',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingBottom: 8,
    borderBottomWidth: 1,
    borderColor: '#eee',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  chapterRow: {
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderColor: '#f0f0f0',
  },
  chapterText: {
    fontSize: 16,
  },
});