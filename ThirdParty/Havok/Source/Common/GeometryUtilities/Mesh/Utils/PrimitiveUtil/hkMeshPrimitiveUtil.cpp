/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Utils/PrimitiveUtil/hkMeshPrimitiveUtil.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>

/* static */hkMeshPrimitiveUtil::PrimitiveStyle HK_CALL hkMeshPrimitiveUtil::getPrimitiveStyle(hkMeshSection::PrimitiveType type)
{
    switch (type)
    {
        default: 
		{
			HK_ASSERT(0xdd827296, !"Unknown type");
			return PRIMITIVE_STYLE_UNKNOWN;
		}
        case hkMeshSection::PRIMITIVE_TYPE_UNKNOWN:			return PRIMITIVE_STYLE_UNKNOWN;
        case hkMeshSection::PRIMITIVE_TYPE_POINT_LIST:		return PRIMITIVE_STYLE_POINT;
        case hkMeshSection::PRIMITIVE_TYPE_LINE_LIST:		return PRIMITIVE_STYLE_LINE;
        case hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST: // fallthru
        case hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_STRIP:	return PRIMITIVE_STYLE_TRIANGLE;
    }
}

/* static */int hkMeshPrimitiveUtil::calculateNumIndices(hkMeshSection::PrimitiveType type, int numPrims)
{
    switch (type)
    {
        case hkMeshSection::PRIMITIVE_TYPE_POINT_LIST:          return numPrims;
        case hkMeshSection::PRIMITIVE_TYPE_LINE_LIST:           return numPrims * 2;
        case hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST:       return numPrims * 3;
        case hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_STRIP:      return numPrims + 2;
        default: break;
    }

    HK_ASSERT(0x31432423, !"Unknown primitive type");
    return 0;
}

/* static */int hkMeshPrimitiveUtil::calculateNumPrimitives(hkMeshSection::PrimitiveType type, int numIndices)
{
    switch (type)
    {
        case hkMeshSection::PRIMITIVE_TYPE_POINT_LIST:          return numIndices;
        case hkMeshSection::PRIMITIVE_TYPE_LINE_LIST:           return numIndices / 2;
        case hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST:       return numIndices / 3;
        case hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_STRIP:      return numIndices - 2;
        default: break;
    }
    HK_ASSERT(0x31432423, !"Unknown primitive type");
    return 0;
}

/* static */void hkMeshPrimitiveUtil::appendTriangleIndices(hkMeshSection::PrimitiveType primType, int numVertices, int indexBase, hkArrayBase<hkUint16>& indicesOut, hkMemoryAllocator& a)
{
    switch (primType)
    {
        case hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST:
        {
            hkUint16* dstIndices = indicesOut._expandBy(a, numVertices);
            for (int i = 0; i < numVertices; i++)
            {
                dstIndices[i] = hkUint16(indexBase + i);
            }
            return;
        }
        case hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_STRIP:
        {
            // Convert
            hkUint16* dstIndices = indicesOut._expandBy(a, (numVertices - 2) * 3);
            for (int i = 2; i < numVertices; ++i)
            {
                *dstIndices++ = hkUint16(indexBase + i - 2);
                if ( (i & 1) == 0)
                {
                    *dstIndices++ = hkUint16(indexBase + i - 1);
                    *dstIndices++ = hkUint16(indexBase + i);
                }
                else
                {
                    *dstIndices++ = hkUint16(indexBase + i);
                    *dstIndices++ = hkUint16(indexBase + i - 1);
                }
            }
            return;
        }
        default:
            HK_ASSERT(0x243432, !"Unhandled type");
    }
}

/* static */void hkMeshPrimitiveUtil::appendTriangleIndices(hkMeshSection::PrimitiveType primType, int numVertices, int indexBase, hkArray<hkUint32>& indicesOut)
{
	switch (primType)
	{
		case hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST:
		{
			hkUint32* dstIndices = indicesOut.expandBy(numVertices);
			for (int i = 0; i < numVertices; i++)
			{
				dstIndices[i] = hkUint16(indexBase + i);
			}
			return;
		}
		case hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_STRIP:
		{
			// Convert
			hkUint32* dstIndices = indicesOut.expandBy( (numVertices - 2) * 3);
			for (int i = 2; i < numVertices; ++i)
			{
				*dstIndices++ = hkUint32(indexBase + i - 2);
				if ( (i & 1) == 0)
				{
					*dstIndices++ = hkUint32(indexBase + i - 1);
					*dstIndices++ = hkUint32(indexBase + i);
				}
				else
				{
					*dstIndices++ = hkUint32(indexBase + i);
					*dstIndices++ = hkUint32(indexBase + i - 1);
				}
			}
			return;
		}
		default:
			HK_ASSERT(0x243432, !"Unhandled type");
	}
}



/* static */void hkMeshPrimitiveUtil::appendTriangleIndices16(hkMeshSection::PrimitiveType primType, const hkUint16* srcIndices, int numIndices, int indexBase, hkArray<hkUint16>& indicesOut)
{
    switch (primType)
    {
        case hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST:
        {
            hkUint16* dstIndices = indicesOut.expandBy(numIndices);

            if (indexBase == 0)
            {
                hkString::memCpy(dstIndices, srcIndices, numIndices * sizeof(hkUint16));
            }
            else
            {
                for (int i = 0; i < numIndices; i++)
                {
                    dstIndices[i] = hkUint16(indexBase + srcIndices[i]);
                }
            }
            return;
        }
        case hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_STRIP:
        {
            // Convert
            hkUint16* dstIndices = indicesOut.expandBy( (numIndices - 2) * 3);
            for (int i = 2; i < numIndices; ++i)
            {
                *dstIndices++ = hkUint16(indexBase + srcIndices[i - 2]);
                if ( (i & 1) == 0)
                {
                    *dstIndices++ = hkUint16(indexBase + srcIndices[i - 1]);
                    *dstIndices++ = hkUint16(indexBase + srcIndices[i]);
                }
                else
                {
                    *dstIndices++ = hkUint16(indexBase + srcIndices[i]);
                    *dstIndices++ = hkUint16(indexBase + srcIndices[i - 1]);
                }
            }
            return;
        }
        default:
            HK_ASSERT(0x243432, !"Unhandled type");
    }
}

/* static */void hkMeshPrimitiveUtil::appendTriangleIndices32(hkMeshSection::PrimitiveType primType, const hkUint32* srcIndices, int numIndices, int indexBase, hkArray<hkUint32>& indicesOut)
{
	switch (primType)
	{
		case hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST:
		{
			hkUint32* dstIndices = indicesOut.expandBy(numIndices);

			if (indexBase == 0)
			{
				hkString::memCpy4(dstIndices, srcIndices, numIndices);
			}
			else
			{
				for (int i = 0; i < numIndices; i++)
				{
					dstIndices[i] = hkUint32(indexBase + srcIndices[i]);
				}
			}
			return;
		}
		case hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_STRIP:
		{
			// Convert
			hkUint32* dstIndices = indicesOut.expandBy( (numIndices - 2) * 3);
			for (int i = 2; i < numIndices; ++i)
			{
				*dstIndices++ = hkUint32(indexBase + srcIndices[i - 2]);
				if ( (i & 1) == 0)
				{
					*dstIndices++ = hkUint32(indexBase + srcIndices[i - 1]);
					*dstIndices++ = hkUint32(indexBase + srcIndices[i]);
				}
				else
				{
					*dstIndices++ = hkUint32(indexBase + srcIndices[i]);
					*dstIndices++ = hkUint32(indexBase + srcIndices[i - 1]);
				}
			}
			return;
		}
		default:
			HK_ASSERT(0x243432, !"Unhandled type");
	}
}


void hkMeshPrimitiveUtil::appendTriangleIndices(const hkMeshSection& section, hkArray<hkUint16>& indicesOut)
{
    switch (section.m_indexType)
    {
        case hkMeshSection::INDEX_TYPE_NONE:
        {
            // I don't need to get any mesh indices
            appendTriangleIndices(section.m_primitiveType, section.m_numIndices, section.m_vertexStartIndex, indicesOut, hkContainerHeapAllocator().get(&indicesOut));
			break;
        }
        case hkMeshSection::INDEX_TYPE_UINT16:
        {
            switch (section.m_primitiveType)
            {
				case hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST: // Fall thru
				case hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_STRIP:
                {
					appendTriangleIndices16(section.m_primitiveType, (const hkUint16*)section.m_indices, section.m_numIndices, 0, indicesOut);
					break;
                }
				default:
                {
					HK_ASSERT(0x3421423, !"Unhandled index type");
					return;
                }
			}
			break;
		}
		case hkMeshSection::INDEX_TYPE_UINT32:
			{
				switch (section.m_primitiveType)
				{
				case hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST: // Fall thru
				case hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_STRIP:
					{
						hkLocalArray<hkUint32> indices32(section.m_numIndices);
						appendTriangleIndices32(section.m_primitiveType, (const hkUint32*)section.m_indices, section.m_numIndices, 0, indices32);
						for (int i =0; i < section.m_numIndices; i++ )
						{
							HK_ON_DEBUG(if (indices32[i] > 0xffff) HK_WARN_ONCE(0x72922222, "mesh section index range overflow!"); )
							indicesOut.pushBack( hkUint16(indices32[i]) );
						}
						break;
					}
				default:
					{
						HK_ASSERT(0x3421423, !"Unhandled index type");
						return;
					}
				}
				break;
			}
		default:
		{
			HK_ASSERT(0x34234, !"Unknown index type");
			return;
        }
    }
}

void hkMeshPrimitiveUtil::appendTriangleIndices(const hkMeshSection& section, hkArray<hkUint32>& indicesOut)
{
	switch (section.m_indexType)
	{
		case hkMeshSection::INDEX_TYPE_NONE:
		{
			// I don't need to get any mesh indices
			appendTriangleIndices(section.m_primitiveType, section.m_numIndices, section.m_vertexStartIndex, indicesOut);
			break;
		}
		case hkMeshSection::INDEX_TYPE_UINT32:
			{
				switch (section.m_primitiveType)
				{
				case hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST: // Fall thru
				case hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_STRIP:
					{
						appendTriangleIndices32(section.m_primitiveType, (const hkUint32*)section.m_indices, section.m_numIndices, 0, indicesOut);
						break;
					}
				default:
					{
						HK_ASSERT(0x3421423, !"Unhandled index type");
						return;
					}
				}
				break;
			}

		case hkMeshSection::INDEX_TYPE_UINT16:
			{
				switch (section.m_primitiveType)
				{
				case hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST: // Fall thru
				case hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_STRIP:
					{
						hkLocalArray<hkUint16> indices16(section.m_numIndices);
						appendTriangleIndices16(section.m_primitiveType, (const hkUint16*)section.m_indices, section.m_numIndices, 0, indices16);
						for (int i =0; i < section.m_numIndices; i++ )
						{
							indicesOut.pushBack( hkUint32(indices16[i]) );
						}
						break;
					}
				default:
					{
						HK_ASSERT(0x3421423, !"Unhandled index type");
						return;
					}
				}
				break;
			}

		
		default:
		{
			HK_ASSERT(0x34234, !"Unknown index type");
			return;
		}
	}
}


void hkMeshPrimitiveUtil::appendTriangleIndices(const hkMeshShape* shape, int sectionIndex, hkArray<hkUint16>& indicesOut)
{
    hkMeshSection section;
	shape->lockSection(sectionIndex, hkMeshShape::ACCESS_INDICES, section);
	appendTriangleIndices(section, indicesOut);
	shape->unlockSection(section);
}

//
//	Merged primitives calculator constructor

hkMergeMeshPrimitvesCalculator::hkMergeMeshPrimitvesCalculator()
:	m_mergedPrimitiveType(hkMeshSection::PRIMITIVE_TYPE_UNKNOWN)
,	m_mergedIndexType(hkMeshSection::INDEX_TYPE_UINT16)
,	m_numTotalPrimitives(0)
{}

//
//	Adds a set of primitives

void hkMergeMeshPrimitvesCalculator::add(int numPrimitives, hkMeshSection::PrimitiveType primtiveType, hkMeshSection::MeshSectionIndexType indexType)
{	
	m_numTotalPrimitives	+= numPrimitives;
	m_mergedIndexType		= (hkMeshSection::MeshSectionIndexType)hkMath::max2(int(m_mergedIndexType), indexType);

	switch ( primtiveType )
	{
	case hkMeshSection::PRIMITIVE_TYPE_POINT_LIST:
		{
			switch ( m_mergedPrimitiveType )
			{
			case hkMeshSection::PRIMITIVE_TYPE_UNKNOWN:
			case hkMeshSection::PRIMITIVE_TYPE_POINT_LIST:		m_mergedPrimitiveType = hkMeshSection::PRIMITIVE_TYPE_POINT_LIST;		break;
			case hkMeshSection::PRIMITIVE_TYPE_LINE_LIST:		m_mergedPrimitiveType = hkMeshSection::PRIMITIVE_TYPE_LINE_LIST;		break;
			default:											m_mergedPrimitiveType = hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST;	break;
			}
		}
		break;

	case hkMeshSection::PRIMITIVE_TYPE_LINE_LIST:
		{
			switch ( m_mergedPrimitiveType )
			{
			case hkMeshSection::PRIMITIVE_TYPE_UNKNOWN:
			case hkMeshSection::PRIMITIVE_TYPE_POINT_LIST:		
			case hkMeshSection::PRIMITIVE_TYPE_LINE_LIST:		m_mergedPrimitiveType = hkMeshSection::PRIMITIVE_TYPE_LINE_LIST;		break;
			default:											m_mergedPrimitiveType = hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST;	break;
			}
		}
		break;

	case hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_STRIP:
		{
			switch ( m_mergedPrimitiveType )
			{
			case hkMeshSection::PRIMITIVE_TYPE_UNKNOWN:
			case hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_STRIP:	m_mergedPrimitiveType = hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_STRIP;	break;
			default:											m_mergedPrimitiveType = hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST;	break;
			}
		}
		break;

//	case hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST:
	default:
		{
			m_mergedPrimitiveType = hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST;
		}
		break;
	}
}

//
//	Copies the indices from source to destination, with conversion

void HK_CALL hkMeshPrimitiveUtil::copyPrimitives(	const void* srcIndexPtr, hkMeshSection::PrimitiveType srcPrimitiveType, hkMeshSection::MeshSectionIndexType srcIndexType,
													int numPrimitives,
													void* dstIndexPtr, hkMeshSection::PrimitiveType dstPrimitiveType, hkMeshSection::MeshSectionIndexType dstIndexType,
													int baseIndexOffset)
{
	PrimitiveProvider src(srcIndexPtr, srcPrimitiveType, srcIndexType);
	PrimitiveProvider dst(dstIndexPtr, dstPrimitiveType, dstIndexType);

	for (int pi = 0; pi < numPrimitives; pi++)
	{
		// Read primitive
		src.readPrimitive();

		// Write it
		dst.writePrimitive(src.m_a + baseIndexOffset, src.m_b + baseIndexOffset, src.m_c + baseIndexOffset);
	}
}

void HK_CALL hkMeshPrimitiveUtil::copyAndRemapPrimitives(	const void* srcIndexPtr, hkMeshSection::PrimitiveType srcPrimitiveType, hkMeshSection::MeshSectionIndexType srcIndexType, int numPrimitives,
															void* dstIndexPtr, hkMeshSection::PrimitiveType dstPrimitiveType, hkMeshSection::MeshSectionIndexType dstIndexType, const int* dstFromSrcIndexRemapTable)
{
	PrimitiveProvider src(srcIndexPtr, srcPrimitiveType, srcIndexType);
	PrimitiveProvider dst(dstIndexPtr, dstPrimitiveType, dstIndexType);

	for (int pi = 0; pi < numPrimitives; pi++)
	{
		// Read primitive
		src.readPrimitive();

		// Write it
		const int vidxA = dstFromSrcIndexRemapTable[src.m_a];
		const int vidxB = dstFromSrcIndexRemapTable[src.m_b];
		const int vidxC = dstFromSrcIndexRemapTable[src.m_c];
		HK_ASSERT(0x70fd2364, (vidxA >= 0) && (vidxB >= 0) && (vidxC >= 0));
		dst.writePrimitive(vidxA, vidxB, vidxC);
	}
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
