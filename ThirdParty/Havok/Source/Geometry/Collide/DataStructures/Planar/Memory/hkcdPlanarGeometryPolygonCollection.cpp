/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Geometry/Collide/hkcdCollide.h>
#include <Geometry/Collide/DataStructures/Planar/Memory/hkcdPlanarGeometryPolygonCollection.h>

//
//	Constructor

hkcdPlanarGeometryPolygonCollection::hkcdPlanarGeometryPolygonCollection()
{
	clear();
}

//
//	Copy constructor

hkcdPlanarGeometryPolygonCollection::hkcdPlanarGeometryPolygonCollection(const hkcdPlanarGeometryPolygonCollection& other)
:	hkcdPlanarGeometryPrimitives::Collection<hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_BIT>()
{
	copy(other);
}

//
//	Compacts the storage

void hkcdPlanarGeometryPolygonCollection::compactStorage()
{
	hkArray<int> newStorage;

	// Copy start block
	newStorage.setSize(MIN_BLOCK_SIZE);
	for (int k = MIN_BLOCK_SIZE - 1; k >= 0; k--)
	{
		newStorage[k] = m_storage[k];
	}

	// Add each valid polygon
	for (PolygonId polyId = getFirstPolygonId(); polyId.isValid(); polyId = getNextPolygonId(polyId))
	{
		const Polygon& srcPoly	= getPolygon(polyId);
		const int numBounds		= getNumBoundaryPlanes(polyId);
		const int size			= 1 + 1 + 1 + 1 + numBounds;
		HK_ASSERT(0x1118cd5, size >= MIN_BLOCK_SIZE);

		// Alloc space for this polygon
		const hkUint32 blockAddr	= newStorage.getSize();
		newStorage.setSize(blockAddr + size, 0);
		Block* dstBlock = reinterpret_cast<Block*>(&newStorage[blockAddr]);
		dstBlock->setSize(size);
		dstBlock->setAllocated();

		// Set it up
		Polygon* dstPoly = reinterpret_cast<Polygon*>(&newStorage[blockAddr]);
		dstPoly->setMaterialId(srcPoly.getMaterialId());
		dstPoly->setNegCellId(srcPoly.getNegCellId());
		dstPoly->setPosCellId(srcPoly.getPosCellId());
		dstPoly->setSupportPlaneId(srcPoly.getSupportPlaneId());

		// Get planes and mark the last one
		int* dstPlanes = reinterpret_cast<int*>(&dstPoly->m_supportId);
		dstPlanes[numBounds] |= END_PAYLOAD_FLAG;
		for (int k = numBounds - 1; k >= 0; k--)
		{
			dstPoly->setBoundaryPlaneId(k, srcPoly.getBoundaryPlaneId(k));
		}
	}

	// Add last block
	{
		int* lastDstBlock = newStorage.expandBy(MIN_BLOCK_SIZE);
		int* lastSrcBlock = &m_storage[m_storage.getSize() - MIN_BLOCK_SIZE];
		for (int k = MIN_BLOCK_SIZE - 1; k >= 0; k--)
		{
			lastDstBlock[k] = lastSrcBlock[k];
		}
	}

	// Clear all bitmaps, no free blocks
	m_primaryBitmap	= 0;
	for (int i = 0; i < NUM_SECONDARY_BMPS; i++)
	{
		m_secondaryBitmaps[i] = 0;
		for (int j = 0; j < MAX_DIVISIONS; j++)
		{
			m_freeBlocks[i][j] = INVALID_BLOCK_ADDR;
		}
	}

	// Replace storage
	m_storage.swap(newStorage);
	m_storage.optimizeCapacity(0);
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
