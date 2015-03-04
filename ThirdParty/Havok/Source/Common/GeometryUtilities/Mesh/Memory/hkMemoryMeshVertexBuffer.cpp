/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Memory/hkMemoryMeshVertexBuffer.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/GeometryUtilities/Mesh/Utils/VertexBufferUtil/hkMeshVertexBufferUtil.h>

hkMemoryMeshVertexBuffer::hkMemoryMeshVertexBuffer(const hkVertexFormat& format, int numVertices)
{
    m_locked = false;
    setVertexFormat(format);
    setNumVerticesAndZero(numVertices);
	m_isBigEndian = HK_ENDIAN_BIG == 1;
}

hkMemoryMeshVertexBuffer::hkMemoryMeshVertexBuffer()
{
    m_locked = false;
    m_numVertices = 0;
    m_isSharable = true;
    m_vertexStride = 0;
	m_isBigEndian = HK_ENDIAN_BIG == 1;
}

hkMemoryMeshVertexBuffer::hkMemoryMeshVertexBuffer( hkFinishLoadedObjectFlag flag )
:hkMeshVertexBuffer(flag)
, m_format(flag)
, m_memory(flag)
{
	if( flag.m_finishing )
	{
		const bool bufferIsBigEndian	= (bool)m_isBigEndian;
		const bool platformIsBigEndian	= (HK_ENDIAN_BIG == 1);
		const bool flipBytes			= platformIsBigEndian ^ bufferIsBigEndian;
		if ( flipBytes )
		{
			handleEndian();
		}
	}	
}

void hkMemoryMeshVertexBuffer::handleEndian()
{
	hkUint8* vertex = m_memory.begin();

	for( int i = 0; i < m_numVertices; ++i )
	{
		// Swap each element in the vertex
		for( int j = 0; j < m_format.m_numElements; ++j )
		{		
			hkUint8* elementBegin = hkAddByteOffset<hkUint8>(vertex, m_elementOffsets[j]);

			for( int k = 0; k < m_format.m_elements[j].m_numValues; ++k )
			{
				switch( m_format.m_elements[j].m_dataType )
				{
				case hkVertexFormat::TYPE_NONE:					
				case hkVertexFormat::TYPE_INT8:				
				case hkVertexFormat::TYPE_UINT8:
					break;

				case hkVertexFormat::TYPE_INT16:
				case hkVertexFormat::TYPE_UINT16:
					{
						hkAlgorithm::swapBytes(elementBegin, sizeof(hkUint16));						
						elementBegin += sizeof(hkUint16);
					}
					break;

				case hkVertexFormat::TYPE_INT32:					
				case hkVertexFormat::TYPE_UINT32:
				case hkVertexFormat::TYPE_UINT8_DWORD:
				case hkVertexFormat::TYPE_ARGB32:
					{
						hkAlgorithm::swapBytes(elementBegin, sizeof(hkUint32));						
						elementBegin += sizeof(hkUint32);
					}
					break;

				case hkVertexFormat::TYPE_FLOAT16:
					{
						hkAlgorithm::swapBytes(elementBegin, sizeof(hkHalf));						
						elementBegin += sizeof(hkHalf);
					}
					break;
				case hkVertexFormat::TYPE_FLOAT32:
					{
						hkAlgorithm::swapBytes(elementBegin, sizeof(hkFloat32));						
						elementBegin += sizeof(hkFloat32);
					}
					break;
				case hkVertexFormat::TYPE_VECTOR4:
					{
						for( int l = 0; l < 4; ++l )
						{
							hkAlgorithm::swapBytes(elementBegin, sizeof(hkReal));							
							elementBegin += sizeof(hkReal);
						}						
					}
					break;
				case hkVertexFormat::TYPE_LAST:
					break;
				}			
			}			
		}

		// Move to next vertex
		vertex += m_vertexStride;
	}	
}

void hkMemoryMeshVertexBuffer::useExternalBuffer(void* data, int numVertices, int bufferSize)
{
    HK_ASSERT(0xaef142, numVertices * m_vertexStride == bufferSize);
	m_memory.setDataUserFree((hkUint8*)data, bufferSize, bufferSize);
}

void hkMemoryMeshVertexBuffer::setVertexFormat(const hkVertexFormat& format)
{
    HK_ASSERT(0x342343, !m_locked);
    HK_ASSERT(0x86bc0144, format.isCanonicalOrder());

    m_format = format;
    m_numVertices = 0;

    // Calculate offsets
    m_vertexStride = calculateElementOffsets(format, m_elementOffsets);
    m_isSharable = format.calculateSharingType() == hkVertexFormat::SHARING_ALL_SHARED;
}

void hkMemoryMeshVertexBuffer::setNumVerticesAndZero(int numVertices)
{
    // Work out the total size
    int totalSize = m_vertexStride * numVertices;
    // Lets align to 16 byte boundaries
    const int memTotalSize = (totalSize + 15) & ~15;

    // Allocate some memory
    m_memory.setSize(memTotalSize);

    // Clear all the memory
    hkString::memClear16(m_memory.begin(), memTotalSize >> 4);
	m_memory.setSizeUnchecked(totalSize);

    m_numVertices = numVertices;
}


/* static */int HK_CALL hkMemoryMeshVertexBuffer::calculateElementOffsets(const hkVertexFormat& format, int offsets[hkVertexFormat::MAX_ELEMENTS])
{
    int offset = 0;
    const int numElements = format.m_numElements;
	int vertexAlignment = 4;

	// First do hkVector4s, as they have the more difficult alignment (for simd)
    for (int i = 0; i < numElements; i++)
    {
        const hkVertexFormat::Element& ele = format.m_elements[i];
		if (ele.m_dataType == hkVertexFormat::TYPE_VECTOR4)
		{
			offsets[i] = offset;
			offset += sizeof(hkVector4) * ele.m_numValues;
			// The vertex as a whole now needs to be simd byte aligned so the next vertices data will be aligned correctly.
			vertexAlignment = HK_REAL_ALIGNMENT;
		}
	}

    for (int i = 0; i < numElements; i++)
    {
        const hkVertexFormat::Element& ele = format.m_elements[i];
		if (ele.m_dataType != hkVertexFormat::TYPE_VECTOR4)
		{
			offsets[i] = offset;
			int elementSize = ele.calculateSize();
			offset += elementSize;
			// All components are 32 bit aligned
			offset = (offset + 3) & ~3;
		}
    }

	// Align the offset
    offset = (offset + vertexAlignment - 1) & ~(vertexAlignment - 1);
    return offset;
}

/* static */int HK_CALL hkMemoryMeshVertexBuffer::calculateElementOffset(const hkVertexFormat& format, hkVertexFormat::ComponentUsage usage, int subUsage)
{
	int offset = 0;
	const int numElements = format.m_numElements;

	// First do hkVector4s, as they have the more difficult alignment (for simd)
	for (int i = 0; i < numElements; i++)
	{
		const hkVertexFormat::Element& ele = format.m_elements[i];
		if (ele.m_usage == usage && ele.m_subUsage == subUsage)
		{
			return offset;
		}
		if (ele.m_dataType == hkVertexFormat::TYPE_VECTOR4)
		{
			offset += sizeof(hkVector4) * ele.m_numValues;
		}
	}

	for (int i = 0; i < numElements; i++)
	{
		const hkVertexFormat::Element& ele = format.m_elements[i];
		if (ele.m_usage == usage && ele.m_subUsage == subUsage)
		{
			return offset;
		}
		if (ele.m_dataType != hkVertexFormat::TYPE_VECTOR4)
		{
			int elementSize = ele.calculateSize();
			offset += elementSize;
			// All components are 32 bit aligned
			offset = (offset + 3) & ~3;
		}
	}
	return -1;
}


hkMemoryMeshVertexBuffer::~hkMemoryMeshVertexBuffer()
{
    HK_ASSERT(0xd8279a06, m_locked == false);
}

hkMeshVertexBuffer* hkMemoryMeshVertexBuffer::clone()
{
    if (m_isSharable)
    {
        // If they are all shared -> then we don't need to copy
        addReference();
        return this;
    }

    // If any are unshared, we need to deep copy
    hkMemoryMeshVertexBuffer* buffer = new hkMemoryMeshVertexBuffer(m_format, m_numVertices);
    // Work out the total size
    int totalSize = m_vertexStride * m_numVertices;
    // Lets align to 16 byte boundaries
    int numQuads = (totalSize + 15) >> 4;

    hkString::memCpy16(buffer->m_memory.begin(), m_memory.begin(), numQuads);

	// Set to the actual size
	buffer->m_memory.setSizeUnchecked(totalSize);

    return buffer;
}

hkMeshVertexBuffer::LockResult hkMemoryMeshVertexBuffer::lock( const LockInput& input, LockedVertices& lockedVerticesOut )
{
    HK_ASSERT(0x7bba546b, m_locked == false);
    HK_ASSERT(0x827daabb, input.m_startVertex >= 0 && input.m_startVertex + input.m_numVertices <= m_numVertices);
    if (m_locked)
    {
		return RESULT_FAILURE;
    }
    getLockedVertices( input.m_startVertex, input.m_numVertices, lockedVerticesOut );
    m_locked = true;
    return RESULT_SUCCESS;
}

void hkMemoryMeshVertexBuffer::getLockedVertices( int startVertex, int numVertices, LockedVertices& lockedVerticesOut )
{
	if (numVertices < 0)
	{
        numVertices = m_numVertices - startVertex;
	}

    hkUint8* vertexStart = m_memory.begin() + (m_vertexStride * startVertex);

    lockedVerticesOut.m_isInterleaved = true;
    lockedVerticesOut.m_numVertices = numVertices;

    const int numElements = m_format.m_numElements;
    lockedVerticesOut.m_numBuffers = numElements;
    for (int i = 0; i < numElements; i++)
    {
        const hkVertexFormat::Element& ele = m_format.m_elements[i];
        LockedVertices::Buffer& buffer = lockedVerticesOut.m_buffers[i];

        buffer.m_start = vertexStart + m_elementOffsets[i];
        buffer.m_stride = m_vertexStride;
        buffer.m_element = ele;
    }
}

hkMeshVertexBuffer::LockResult hkMemoryMeshVertexBuffer::partialLock( const LockInput& input, const PartialLockInput& partialInput, LockedVertices& lockedOut)
{
    HK_ASSERT(0x7bba546a, m_locked == false);
    HK_ASSERT(0x827eaabb, input.m_startVertex >= 0 && input.m_startVertex + input.m_numVertices <= m_numVertices);
    if (m_locked)
    {
        return RESULT_FAILURE;
    }

	int numVertices = input.m_numVertices;
	if (numVertices < 0)
	{
		numVertices = m_numVertices - input.m_startVertex;
	}

    hkUint8* vertexStart = m_memory.begin() + (m_vertexStride * input.m_startVertex);

    lockedOut.m_isInterleaved = true;
    lockedOut.m_numVertices = numVertices;

    const int numBuffers = partialInput.m_numLockFlags;
    lockedOut.m_numBuffers = numBuffers;

    for (int i = 0; i < numBuffers; i++)
    {
        int elementIndex = partialInput.m_elementIndices[i];

        HK_ASSERT(0xd8279a35, elementIndex >= 0 && elementIndex < m_format.m_numElements);

        const hkVertexFormat::Element& ele = m_format.m_elements[elementIndex];
        LockedVertices::Buffer& buffer = lockedOut.m_buffers[i];

        // Write it out
        buffer.m_start = vertexStart + m_elementOffsets[elementIndex];
        buffer.m_stride = m_vertexStride;
        buffer.m_element = ele;
    }

    m_locked = true;
    return RESULT_SUCCESS;
}

void hkMemoryMeshVertexBuffer::getLockedVerticesBuffer(int elementIndex, LockedVertices::Buffer& buffer)
{
    const hkVertexFormat::Element& ele = m_format.m_elements[elementIndex];
    // Write it out
    buffer.m_start = m_memory.begin() + m_elementOffsets[elementIndex];
    buffer.m_stride = m_vertexStride;
    buffer.m_element = ele;
}

void hkMemoryMeshVertexBuffer::unlock(const LockedVertices& lockedVertices )
{
    HK_ASSERT(0xbd838ddd, m_locked);
    m_locked = false;
}

void hkMemoryMeshVertexBuffer::getElementVectorArray(const LockedVertices& lockedVertices, int elementIndex, hkFloat32* data)
{
	hkMeshVertexBufferUtil::getElementVectorArray(lockedVertices, elementIndex, data);
}

void hkMemoryMeshVertexBuffer::setElementVectorArray(const LockedVertices& lockedVertices, int elementIndex, const hkFloat32* data)
{
	hkMeshVertexBufferUtil::setElementVectorArray(lockedVertices, elementIndex, data);
}

void hkMemoryMeshVertexBuffer::getElementIntArray(const LockedVertices& lockedVertices, int elementIndex, int* data)
{
    hkMeshVertexBufferUtil::getElementIntArray(lockedVertices, elementIndex, data);
}

void hkMemoryMeshVertexBuffer::setElementIntArray(const LockedVertices& lockedVertices, int elementIndex, const int* data)
{
    hkMeshVertexBufferUtil::setElementIntArray(lockedVertices, elementIndex, data);
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
