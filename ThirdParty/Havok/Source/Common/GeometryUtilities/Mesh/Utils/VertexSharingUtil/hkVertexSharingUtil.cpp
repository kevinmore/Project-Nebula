/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Utils/VertexSharingUtil/hkVertexSharingUtil.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/GeometryUtilities/Mesh/Memory/hkMemoryMeshVertexBuffer.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>

//
//	Constructor. Initializes all thresholds to 1.0e-6f

hkVertexSharingUtil::Threshold::Threshold(hkReal defaultThreshold)
{
	setAll(defaultThreshold);
}

//
//	Constructor.

hkVertexSharingUtil::hkVertexSharingUtil()
:	m_threshold(hkReal(1.0e-4f))
{
    m_lockedWorkVertex.m_isInterleaved = true;
    m_lockedWorkVertex.m_numVertices = 1;
}

void hkVertexSharingUtil::begin(const hkVertexFormat& format, int numVerticesEstimate)
{
    HK_ASSERT(0x242342, format.isCanonicalOrder());

    m_format = format;

    m_hashMap.clear(); m_hashMap.reserve(numVerticesEstimate);
    m_entries.clear(); m_entries.reserve(numVerticesEstimate);

    // Work out the layout
    m_vertexStride = hkMemoryMeshVertexBuffer::calculateElementOffsets(format, m_elementOffsets);

    m_workVertex.setSize(m_vertexStride);
    hkString::memSet(m_workVertex.begin(), 0, m_vertexStride);

    m_lockedWorkVertex.m_numBuffers = format.m_numElements;

    // Set up the layout
    {
        const int numElements = format.m_numElements;
        for (int i = 0; i < numElements; i++)
        {
            hkMeshVertexBuffer::LockedVertices::Buffer& buffer = m_lockedWorkVertex.m_buffers[i];

            buffer.m_start = m_workVertex.begin() + m_elementOffsets[i];
            buffer.m_stride = m_vertexStride;
            buffer.m_element = format.m_elements[i];
        }
    }

    m_numVertices = 0;
    m_vertices.clear(); m_vertices.reserve(numVerticesEstimate * m_vertexStride);
}


bool hkVertexSharingUtil::isVertexExactlyEqual(const hkUint8* a, const hkUint8* b) const
{
    // We will just do a memory compare.
    // We know must be dword aligned
    HK_ASSERT(0x3423432, (m_vertexStride % 4) == 0);

    const int numWords = m_vertexStride / 4;
    const hkUint32* wordsA = (const hkUint32*)a;
    const hkUint32* wordsB = (const hkUint32*)b;

    for (int i = 0; i < numWords; i++)
    {
        if (wordsA[i] != wordsB[i])
        {
            return false;
        }
    }

    return true;
}

//
//	Returns true if the element is a float32 direction, i.e. normal, binormal or tangent

static HK_FORCE_INLINE bool HK_CALL _isDirection(const hkVertexFormat::Element& e)
{
	return	(e.m_dataType == hkVertexFormat::TYPE_FLOAT32)		&&
			(e.m_numValues == 3)								&&
			(	(e.m_usage == hkVertexFormat::USAGE_NORMAL)		||
				(e.m_usage == hkVertexFormat::USAGE_BINORMAL)	||
				(e.m_usage == hkVertexFormat::USAGE_TANGENT));
}

bool hkVertexSharingUtil::isVertexEqual(const hkUint8* a, const hkUint8* b) const
{
    if (isVertexExactlyEqual(a, b))
    {
        return true;
    }

	const int numElements = m_format.m_numElements;

	for (int i = 0; i < numElements; i++)
	{
		const hkVertexFormat::Element& ele = m_format.m_elements[i];
		const int numValues = ele.m_numValues;

		// Get element threshold
		const hkSimdReal threshold = hkSimdReal::fromFloat(getThreshold(ele.m_usage));
		if ( threshold.isLessEqualZero() )
		{
			// Must be exactly the same if the threshold is 0
			return false;
		}

		// If this is a direction (i.e. normal, binormal, tangent) use (angle, magnitude) tolerance
		if ( _isDirection(ele) )
		{
			const hkFloat32* ptra = (const hkFloat32*)(a + m_elementOffsets[i]);
			const hkFloat32* ptrb = (const hkFloat32*)(b + m_elementOffsets[i]);

			hkVector4 vA;	vA.load<3,HK_IO_NATIVE_ALIGNED>(ptra);
			hkVector4 vB;	vB.load<3,HK_IO_NATIVE_ALIGNED>(ptrb);

			// Compute dot and magnitudes
			hkVector4 vDots;	hkVector4Util::dot3_3vs3(vA, vB, vA, vA, vB, vB, vDots);
			hkVector4 vLen;		vLen.setSqrt(vDots);

			const hkSimdReal lenA	= vLen.getComponent<1>();		// Length(vA)
			const hkSimdReal lenB	= vLen.getComponent<2>();		// Length(vB)
			const hkSimdReal lenAB	= lenA * lenB;					// Length(vA) * Length(vB)
			const hkSimdReal dotAB	= vDots.getComponent<0>();		// Dot(vA, vB)
			const hkSimdReal relErr	= lenAB - dotAB;				// Length(vA) * Length(vB) - Dot(vA, vB)
			hkSimdReal lenErr;		lenErr.setAbs(lenA - lenB);

			const hkSimdReal relTol = lenAB * hkSimdReal::fromFloat(m_threshold.m_angularThreshold);
			if ( (relErr > relTol) || (lenErr > threshold) )
			{
				return false;
			}
		}
		else
		{
			// See if they are equal
			switch ( ele.m_dataType )
			{
			case hkVertexFormat::TYPE_FLOAT32:
				{
					const hkFloat32* va = (const hkFloat32*)(a + m_elementOffsets[i]);
					const hkFloat32* vb = (const hkFloat32*)(b + m_elementOffsets[i]);
					for (int j = 0; j < numValues; j++)
					{
						hkSimdReal vaj; vaj.load<1>(&va[j]);
						hkSimdReal vbj; vbj.load<1>(&vb[j]);
						hkSimdReal absdiff; absdiff.setAbs(vaj - vbj);
						if (absdiff.isGreater(threshold))
						{
							return false;
						}
					}
					break;
				}
			case hkVertexFormat::TYPE_ARGB32:
			case hkVertexFormat::TYPE_INT32:
			case hkVertexFormat::TYPE_UINT32:
			case hkVertexFormat::TYPE_UINT8_DWORD:
				{
					const hkUint32* va = (const hkUint32*)(a + m_elementOffsets[i]);
					const hkUint32* vb = (const hkUint32*)(b + m_elementOffsets[i]);
					for (int j = 0; j < numValues; j++)
					{
						if (va[j] != vb[j])
						{
							return false;
						}
					}
					break;

				}
			case hkVertexFormat::TYPE_INT8:
			case hkVertexFormat::TYPE_UINT8:
				{
					const hkUint8* va = (const hkUint8*)(a + m_elementOffsets[i]);
					const hkUint8* vb = (const hkUint8*)(b + m_elementOffsets[i]);
					for (int j = 0; j < numValues; j++)
					{
						if (va[j] != vb[j])
						{
							return false;
						}
					}
					break;
				}
			case hkVertexFormat::TYPE_FLOAT16:
			case hkVertexFormat::TYPE_INT16:
			case hkVertexFormat::TYPE_UINT16:
				{
					const hkUint16* va = (const hkUint16*)(a + m_elementOffsets[i]);
					const hkUint16* vb = (const hkUint16*)(b + m_elementOffsets[i]);
					for (int j = 0; j < numValues; j++)
					{
						if (va[j] != vb[j])
						{
							return false;
						}
					}
					break;
				}

			default:
				{
					HK_ASSERT(0xa01ef142, !"Unknown type");
					return false;
				}
			}
		}
	}
	return true;
}

int hkVertexSharingUtil::findVertexIndex(hkUint32 hash, const void* vertex) const
{
    hkPointerMap<hkUint32, int>::Iterator iter = m_hashMap.findKey(hash);

    const hkUint8* workVertex = reinterpret_cast<const hkUint8*>(vertex);
    if (m_hashMap.isValid(iter))
    {
		const int oldEntryIndex = m_hashMap.getValue(iter);
        const Entry* curEntry = &m_entries[oldEntryIndex];

        while (true)
        {
            // See if we have found a match
            const hkUint8* testVertex = m_vertices.begin() + m_vertexStride * curEntry->m_vertexIndex;
            if (isVertexEqual(workVertex, testVertex))
            {
				// Found
                return curEntry->m_vertexIndex;
            }
			const int nextIndex = curEntry->m_nextEntryIndex;
            if ( nextIndex < 0)
            {
                break;
            }
            // Next
            curEntry = &m_entries[nextIndex];
        }
    }
    // Not found
    return -1;
}

int hkVertexSharingUtil::addVertex(hkUint32 hash, const void* vertex)
{
	const int entryIndex = m_entries.getSize();

	const hkPointerMap<hkUint32, int>::Iterator iter = m_hashMap.findKey(hash);

    const hkUint8* workVertex = reinterpret_cast<const hkUint8*>(vertex);
    if (m_hashMap.isValid(iter))
    {
		const int oldEntryIndex = m_hashMap.getValue(iter);
        Entry* curEntry = &m_entries[oldEntryIndex];

        while (true)
        {
            // See if we have found a match
            const hkUint8* testVertex = m_vertices.begin() + m_vertexStride * curEntry->m_vertexIndex;
            if (isVertexEqual(workVertex, testVertex))
            {
				// Found
                return curEntry->m_vertexIndex;
            }

			const int nextIndex = curEntry->m_nextEntryIndex;
            if ( nextIndex < 0)
            {
                break;
            }
            // Next
            curEntry = &m_entries[nextIndex];
        }

        // Not found
		curEntry->m_nextEntryIndex = entryIndex;
	}
    else
    {
        // Not found -> so add a new entry
        m_hashMap.insert(hash, entryIndex);
	}

	Entry& entry = m_entries.expandOne();
    const int vertexIndex = m_numVertices;

    // We need to add a new vertex
    m_vertices.setSize(m_vertices.getSize() + m_vertexStride);
    m_numVertices++;
    // Copy the vertex over
    hkString::memCpy(m_vertices.end() - m_vertexStride, workVertex, m_vertexStride);

    // Add entry
    entry.m_nextEntryIndex = -1;
    entry.m_vertexIndex = vertexIndex;

    return vertexIndex;
}

int hkVertexSharingUtil::addVertex(hkUint32 hash)
{
    return addVertex(hash, m_workVertex.begin());
}

int hkVertexSharingUtil::end(hkMeshVertexBuffer::LockedVertices& lockedVertices)
{
    lockedVertices.m_numVertices = m_numVertices;
    lockedVertices.m_isInterleaved = true;
    lockedVertices.m_numBuffers = m_format.m_numElements;

    const int numElements = m_format.m_numElements;
    for (int i = 0; i < numElements; i++)
    {
        hkMeshVertexBuffer::LockedVertices::Buffer& buffer = lockedVertices.m_buffers[i];

        buffer.m_start = m_vertices.begin() + m_elementOffsets[i];
        buffer.m_stride = m_vertexStride;
        buffer.m_element = m_format.m_elements[i];
    }

    return m_numVertices;
}

//
//	Sets all thresholds to the given value

void hkVertexSharingUtil::Threshold::setAll(hkReal threshold)
{
	m_angularThreshold	= threshold;

	for (int k = 0; k < hkVertexFormat::USAGE_LAST; k++)
	{
		m_thresholds[k] = threshold;
	}
}

//
//	Sets the thresholds

void hkVertexSharingUtil::setThresholds(const Threshold& t)
{
	m_threshold = t;
}

//
//	Sets the given component threshold

void hkVertexSharingUtil::Threshold::set(hkVertexFormat::ComponentUsage componentUsage, hkReal threshold)
{
	HK_ASSERT(0x294c7150, (componentUsage >= 0) && (componentUsage < hkVertexFormat::USAGE_LAST));
	m_thresholds[componentUsage] = threshold;
}

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
