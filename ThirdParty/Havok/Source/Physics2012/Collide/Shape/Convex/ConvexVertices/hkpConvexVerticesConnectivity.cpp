/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

// This
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesConnectivity.h>

#include <Common/Base/Container/PointerMap/hkPointerMap.h>

void hkpConvexVerticesConnectivity::clear()
{
    m_vertexIndices.clear();
    m_numVerticesPerFace.clear();
}

void hkpConvexVerticesConnectivity::addFace(int* indices,int numIndices)
{
    HK_ASSERT(0x32424234,numIndices<256 && numIndices>=3);

    // add the indices
    m_numVerticesPerFace.pushBack(hkUint8(numIndices));

    // Add all of the indices
    for (int i = 0; i < numIndices; i++)
    {
        m_vertexIndices.pushBack(hkUint16(indices[i]));
    }
}


hkBool hkpConvexVerticesConnectivity::isClosed() const
{
	// For the thing to be closed, every edge must have a pair in the opposite direction
	// So we go through all the faces looking - ensuring every edge occurs once in each direction

	// Key is 16/16 bit concat of the indices. To make edges unique the order always goes from smaller to
    // larger index
    hkPointerMap<hkUint32,hkUint32> edgeMap;

	const int numFaces = getNumFaces();
    int faceStart = 0;
    for (int i = 0; i < numFaces; i++)
    {
        const int numFaceIndices = m_numVerticesPerFace[i];

		const hkUint16* indices = &(m_vertexIndices[faceStart]);
		HK_ASSERT(0x34234,faceStart + numFaceIndices <= m_vertexIndices.getSize());

		int startIndex = indices[numFaceIndices-1];
		
        for (int j = 0; j < numFaceIndices; j++)
        {
			int endIndex = indices[j];
			// Work out the edge index

			hkUint32 key = ((startIndex<endIndex)?(startIndex<<16)|(endIndex):(endIndex<<16)|(startIndex))+1;
			hkUint32 bit = (startIndex<endIndex)?2:1;
            
            // Do the lookup

            hkPointerMap<hkUint32,hkUint32>::Iterator iter = edgeMap.findKey(key);
            if (edgeMap.isValid(iter))
            {
				hkUint32 bits = edgeMap.getValue(iter);
                if ((bits&bit) != 0)
				{
					// If the bit is set then this has already been seen -> which would be an error
					return false;
				}
				bits |= bit;
				edgeMap.setValue(iter,bits);
            }
            else
            {
				edgeMap.insert(key,bit);
            }

			startIndex = endIndex;
        }

        faceStart += numFaceIndices;
    }

	// If we iterate through the edge combinations all should have both bits set

	hkPointerMap<hkUint32,hkUint32>::Iterator iter = edgeMap.getIterator();
	while (edgeMap.isValid(iter))
	{
		// Get the value 
		hkUint32 bits = edgeMap.getValue(iter);

		if (bits != 3)
		{
			//hkUint32 key = edgeMap.getKey(iter);

			//int startIndex = (key>>16)&0xffff;
			//int endIndex = (key&0xffff)-1;

			// All should be set...
			return false;
		}

		// next
		iter = edgeMap.getNext(iter);
	}
    
	return true;
}

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
