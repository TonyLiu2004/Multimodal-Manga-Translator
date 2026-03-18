import { useLocalSearchParams } from 'expo-router';
import { useEffect, useState } from 'react';
import { ActivityIndicator, View } from 'react-native';
import resolveAssetSource from 'react-native/Libraries/Image/resolveAssetSource';
import MangaReader from '../components/MangaReader'; 

const IS_TESTING = true; // true for testing with local images

export default function ReaderScreen() {
    const { id } = useLocalSearchParams();
    const [pages, setPages] = useState<string[]>([]);
    const [loading, setLoading] = useState(true);

    const runTest = async () => {
        const testPages = [
            resolveAssetSource(require('../../assets/images/test_1.png')).uri,
            resolveAssetSource(require('../../assets/images/test_7.png')).uri,
            resolveAssetSource(require('../../assets/images/cntest_1.png')).uri,
            resolveAssetSource(require('../../assets/images/krtest_1.png')).uri,
        ];

        setPages(testPages);
    };

    // Placeholder
    const runBackend = async () => {}

    useEffect(() => {
        const fetchPages = async () => {
            setLoading(true);
            try {
                if (IS_TESTING) {
                    await runTest();
                } else {
                    await runBackend();
                }
            } catch (error) {
                console.error("Failed to fetch chapter pages:", error);
            } finally {
                setLoading(false);
            }
        };

        if (id) {
            fetchPages();
        }
    }, [id]);

    if (loading) {
        return (
            <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
                <ActivityIndicator size="large" color="#007AFF" />
            </View>
        );
    }

    return <MangaReader pages={pages} />; 
}