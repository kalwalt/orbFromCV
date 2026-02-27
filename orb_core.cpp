#include "orb_core.hpp"
#include <random>

const float HARRIS_K = 0.04f;
const float ORB_PI = 3.14159265358979323846f;

// Helper to replace OpenCV's cvRound
inline int stdRound(float value) {
    return static_cast<int>(std::lround(value));
}

// Helper to replace OpenCV's fastAtan2 (Returns angle in degrees 0..360)
inline float fastAtan2_deg(float y, float x) {
    float angle = std::atan2(y, x) * (180.0f / ORB_PI);
    if (angle < 0.0f) angle += 360.0f;
    return angle;
}

/**
 * Computes the Harris responses in a blockSize x blockSize patch 
 * at given points in the image.
 */
void HarrisResponses(const Image8U& img, const std::vector<Rect>& layerinfo,
                     std::vector<KeyPoint>& pts, int blockSize, float harris_k)
{
    size_t ptidx, ptsize = pts.size();

    const uint8_t* ptr00 = img.ptr();
    size_t size_t_step = img.step;
    int step = static_cast<int>(size_t_step);

    int r = blockSize / 2;

    float scale = 1.f / ((1 << 2) * blockSize * 255.f);
    float scale_sq_sq = scale * scale * scale * scale;

    std::vector<int> ofsbuf(blockSize * blockSize);
    int* ofs = ofsbuf.data();
    for( int i = 0; i < blockSize; i++ )
        for( int j = 0; j < blockSize; j++ )
            ofs[i * blockSize + j] = (int)(i * step + j);

    // This loop is an excellent candidate for WASM SIMD optimization
    for( ptidx = 0; ptidx < ptsize; ptidx++ )
    {
        int x0 = stdRound(pts[ptidx].pt.x);
        int y0 = stdRound(pts[ptidx].pt.y);
        int z = pts[ptidx].octave;

        const uint8_t* ptr0 = ptr00 + (y0 - r + layerinfo[z].y) * size_t_step + (x0 - r + layerinfo[z].x);
        int a = 0, b = 0, c = 0;

        for( int k = 0; k < blockSize * blockSize; k++ )
        {
            const uint8_t* ptr = ptr0 + ofs[k];
            int Ix = (ptr[1] - ptr[-1]) * 2 + (ptr[-step + 1] - ptr[-step - 1]) + (ptr[step + 1] - ptr[step - 1]);
            int Iy = (ptr[step] - ptr[-step]) * 2 + (ptr[step - 1] - ptr[-step - 1]) + (ptr[step + 1] - ptr[-step + 1]);
            a += Ix * Ix;
            b += Iy * Iy;
            c += Ix * Iy;
        }
        pts[ptidx].response = ((float)a * b - (float)c * c -
                               harris_k * ((float)a + b) * ((float)a + b)) * scale_sq_sq;
    }
}

/**
 * Computes the orientation (angle) of the keypoints.
 */
void ICAngles(const Image8U& img, const std::vector<Rect>& layerinfo,
              std::vector<KeyPoint>& pts, const std::vector<int>& u_max, int half_k)
{
    int step = img.step;
    size_t ptidx, ptsize = pts.size();

    for( ptidx = 0; ptidx < ptsize; ptidx++ )
    {
        const Rect& layer = layerinfo[pts[ptidx].octave];
        const uint8_t* center = &img.at(stdRound(pts[ptidx].pt.y) + layer.y, stdRound(pts[ptidx].pt.x) + layer.x);

        int m_01 = 0, m_10 = 0;

        // Treat the center line differently, v=0
        for (int u = -half_k; u <= half_k; ++u)
            m_10 += u * center[u];

        // Go line by line in the circular patch
        for (int v = 1; v <= half_k; ++v)
        {
            // Proceed over the two lines
            int v_sum = 0;
            int d = u_max[v];
            for (int u = -d; u <= d; ++u)
            {
                int val_plus = center[u + v * step], val_minus = center[u - v * step];
                v_sum += (val_plus - val_minus);
                m_10 += u * (val_plus + val_minus);
            }
            m_01 += v * v_sum;
        }

        pts[ptidx].angle = fastAtan2_deg((float)m_01, (float)m_10);
    }
}

/**
 * Computes the ORB descriptors for the given keypoints.
 */
void computeOrbDescriptors(const Image8U& imagePyramid, const std::vector<Rect>& layerInfo,
                           const std::vector<float>& layerScale, std::vector<KeyPoint>& keypoints,
                           Image8U& descriptors, const std::vector<Point2i>& _pattern, int dsize, int wta_k)
{
    int step = imagePyramid.step;
    int j, i, nkeypoints = (int)keypoints.size();

    for( j = 0; j < nkeypoints; j++ )
    {
        const KeyPoint& kpt = keypoints[j];
        const Rect& layer = layerInfo[kpt.octave];
        float scale = 1.f / layerScale[kpt.octave];
        float angle = kpt.angle;

        angle *= (float)(ORB_PI / 180.f);
        float a = std::cos(angle), b = std::sin(angle);

        const uint8_t* center = &imagePyramid.at(stdRound(kpt.pt.y * scale) + layer.y,
                                                 stdRound(kpt.pt.x * scale) + layer.x);
        float x, y;
        int ix, iy;
        const Point2i* pattern = &_pattern[0];
        uint8_t* desc = descriptors.ptr(j);

        // Macro equivalent replaced with inline logic for clarity and porting
        #define GET_VALUE(idx) \
               (x = pattern[idx].x * a - pattern[idx].y * b, \
                y = pattern[idx].x * b + pattern[idx].y * a, \
                ix = stdRound(x), \
                iy = stdRound(y), \
                *(center + iy * step + ix) )

        if( wta_k == 2 )
        {
            for (i = 0; i < dsize; ++i, pattern += 16)
            {
                int t0, t1, val;
                t0 = GET_VALUE(0); t1 = GET_VALUE(1);
                val = t0 < t1;
                t0 = GET_VALUE(2); t1 = GET_VALUE(3);
                val |= (t0 < t1) << 1;
                t0 = GET_VALUE(4); t1 = GET_VALUE(5);
                val |= (t0 < t1) << 2;
                t0 = GET_VALUE(6); t1 = GET_VALUE(7);
                val |= (t0 < t1) << 3;
                t0 = GET_VALUE(8); t1 = GET_VALUE(9);
                val |= (t0 < t1) << 4;
                t0 = GET_VALUE(10); t1 = GET_VALUE(11);
                val |= (t0 < t1) << 5;
                t0 = GET_VALUE(12); t1 = GET_VALUE(13);
                val |= (t0 < t1) << 6;
                t0 = GET_VALUE(14); t1 = GET_VALUE(15);
                val |= (t0 < t1) << 7;

                desc[i] = (uint8_t)val;
            }
        }
        else if( wta_k == 3 )
        {
            for (i = 0; i < dsize; ++i, pattern += 12)
            {
                int t0, t1, t2, val;
                t0 = GET_VALUE(0); t1 = GET_VALUE(1); t2 = GET_VALUE(2);
                val = t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0);

                t0 = GET_VALUE(3); t1 = GET_VALUE(4); t2 = GET_VALUE(5);
                val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 2;

                t0 = GET_VALUE(6); t1 = GET_VALUE(7); t2 = GET_VALUE(8);
                val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 4;

                t0 = GET_VALUE(9); t1 = GET_VALUE(10); t2 = GET_VALUE(11);
                val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 6;

                desc[i] = (uint8_t)val;
            }
        }
        else if( wta_k == 4 )
        {
            for (i = 0; i < dsize; ++i, pattern += 16)
            {
                int t0, t1, t2, t3, u, v, k, val;
                
                t0 = GET_VALUE(0); t1 = GET_VALUE(1);
                t2 = GET_VALUE(2); t3 = GET_VALUE(3);
                u = 0, v = 2;
                if( t1 > t0 ) t0 = t1, u = 1;
                if( t3 > t2 ) t2 = t3, v = 3;
                k = t0 > t2 ? u : v;
                val = k;

                t0 = GET_VALUE(4); t1 = GET_VALUE(5);
                t2 = GET_VALUE(6); t3 = GET_VALUE(7);
                u = 0, v = 2;
                if( t1 > t0 ) t0 = t1, u = 1;
                if( t3 > t2 ) t2 = t3, v = 3;
                k = t0 > t2 ? u : v;
                val |= k << 2;

                t0 = GET_VALUE(8); t1 = GET_VALUE(9);
                t2 = GET_VALUE(10); t3 = GET_VALUE(11);
                u = 0, v = 2;
                if( t1 > t0 ) t0 = t1, u = 1;
                if( t3 > t2 ) t2 = t3, v = 3;
                k = t0 > t2 ? u : v;
                val |= k << 4;

                t0 = GET_VALUE(12); t1 = GET_VALUE(13);
                t2 = GET_VALUE(14); t3 = GET_VALUE(15);
                u = 0, v = 2;
                if( t1 > t0 ) t0 = t1, u = 1;
                if( t3 > t2 ) t2 = t3, v = 3;
                k = t0 > t2 ? u : v;
                val |= k << 6;

                desc[i] = (uint8_t)val;
            }
        }
        #undef GET_VALUE
    }
}

// Replaces OpenCV's RNG with standard C++ <random>
void initializeOrbPattern(const Point2i* pattern0, std::vector<Point2i>& pattern, int ntuples, int tupleSize, int poolSize)
{
    std::mt19937 rng(0x12345678); 
    std::uniform_int_distribution<int> dist(0, poolSize - 1);
    
    int i, k, k1;
    pattern.resize(ntuples * tupleSize);

    for( i = 0; i < ntuples; i++ )
    {
        for( k = 0; k < tupleSize; k++ )
        {
            for(;;)
            {
                int idx = dist(rng);
                Point2i pt = pattern0[idx];
                for( k1 = 0; k1 < k; k1++ )
                    if( pattern[tupleSize * i + k1] == pt )
                        break;
                if( k1 == k )
                {
                    pattern[tupleSize * i + k] = pt;
                    break;
                }
            }
        }
    }
}