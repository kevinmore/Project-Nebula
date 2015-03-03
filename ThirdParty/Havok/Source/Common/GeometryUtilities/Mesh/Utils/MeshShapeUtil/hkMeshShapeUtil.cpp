/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Utils/MeshShapeUtil/hkMeshShapeUtil.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Container/BitField/hkBitField.h>
#include <Common/GeometryUtilities/Mesh/hkMeshMaterialRegistry.h>
#include <Common/GeometryUtilities/Mesh/Utils/FindUniquePositionsUtil/hkFindUniquePositionsUtil.h>
#include <Common/GeometryUtilities/Mesh/Utils/MeshMaterialUtil/hkMeshMaterialUtil.h>
#include <Common/GeometryUtilities/Mesh/Utils/MeshSectionLockSet/hkMeshSectionLockSet.h>
#include <Common/GeometryUtilities/Mesh/Utils/VertexBufferUtil/hkMeshVertexBufferUtil.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabb.h>
#include <Common/Base/Math/Vector/hkIntVector.h>

#include <Common/GeometryUtilities/Mesh/Utils/PrimitiveUtil/hkMeshPrimitiveUtil.h>


/* static */ void hkMeshShapeUtil::findUniqueVertexPositions(hkMeshShape* shape, hkArray<hkVector4>& verticesOut)
{
    hkFindUniquePositionsUtil posUtil;

	hkMeshSectionLockSet sectionSet;
	sectionSet.addMeshSections(shape, hkMeshShape::ACCESS_VERTEX_BUFFER | hkMeshShape::ACCESS_INDICES);

	hkArray<hkMeshVertexBuffer*> vertexBuffers;
	sectionSet.findUniqueVertexBuffers(vertexBuffers);

    const int numVertexBuffers = vertexBuffers.getSize();

	{
		hkArray<hkVector4> vertices;
		hkArray<hkFloat32> va;
		for (int i = 0; i < numVertexBuffers; i++)
		{
			hkMeshVertexBuffer* vertexBuffer = vertexBuffers[i];

			// Get the positions
			hkVertexFormat vertexFormat;
			vertexBuffer->getVertexFormat(vertexFormat);

			int elementIndex = vertexFormat.findElementIndex(hkVertexFormat::USAGE_POSITION, 0);
			if (elementIndex < 0)
			{
				continue;
			}

			//
			hkMeshVertexBuffer::LockInput lockInput;
			hkMeshVertexBuffer::PartialLockInput partialLockInput;

			partialLockInput.m_numLockFlags = 1;
			partialLockInput.m_elementIndices[0] = elementIndex;
			partialLockInput.m_lockFlags[0] = hkMeshVertexBuffer::ACCESS_READ | hkMeshVertexBuffer::ACCESS_ELEMENT_ARRAY;

			hkMeshVertexBuffer::LockedVertices lockedVertices;
			HK_ON_DEBUG(hkMeshVertexBuffer::LockResult lockRes = vertexBuffer->partialLock( lockInput, partialLockInput, lockedVertices));
			HK_ASSERT(0x342432a, lockRes == hkMeshVertexBuffer::RESULT_SUCCESS);

			// Get the positions

			vertices.setSize(lockedVertices.m_numVertices);
			va.setSize(4*lockedVertices.m_numVertices);
			vertexBuffer->getElementVectorArray(lockedVertices, 0, va.begin());
			vertexBuffer->unlock(lockedVertices);
			for (int v=0; v<lockedVertices.m_numVertices; ++v)
			{
				vertices[v].load<4,HK_IO_NATIVE_ALIGNED>(&va[4*v]);
			}

			posUtil.addPositions(vertices.begin(), vertices.getSize());
		}
	}

    verticesOut.swap(posUtil.m_positions);
}

/* static */hkMeshShape* hkMeshShapeUtil::convert(hkMeshMaterialRegistry* srcRegistry, const hkMeshShape* srcShape, hkMeshMaterialRegistry* dstRegistry, hkMeshSystem* dstSystem)
{
	hkMeshSectionLockSet sectionSet;
    sectionSet.addMeshSections(srcShape, hkMeshShape::ACCESS_VERTEX_BUFFER | hkMeshShape::ACCESS_INDICES);

    hkArray<hkMeshVertexBuffer*> srcVertexBuffers;
    sectionSet.findUniqueVertexBuffers(srcVertexBuffers);

    hkArray<hkMeshVertexBuffer*> dstVertexBuffers;
    hkPointerMap<hkMeshVertexBuffer*, hkMeshVertexBuffer*> bufferMap;

    const int numBuffers = srcVertexBuffers.getSize();
    for (int i = 0; i < numBuffers; i++)
    {
        hkMeshVertexBuffer* srcVertexBuffer = srcVertexBuffers[i];

        hkVertexFormat srcVertexFormat;
        srcVertexBuffer->getVertexFormat(srcVertexFormat);

        hkVertexFormat dstVertexFormat;
        dstSystem->findSuitableVertexFormat(srcVertexFormat, dstVertexFormat);

        // Create the dst vertex buffer
        hkMeshVertexBuffer* dstVertexBuffer = dstSystem->createVertexBuffer(dstVertexFormat, srcVertexBuffer->getNumVertices());

        // Convert all the src stuff over
        hkMeshVertexBufferUtil::convert(srcVertexBuffer, dstVertexBuffer);

        // Add to the map
        bufferMap.insert(srcVertexBuffer, dstVertexBuffer);
        dstVertexBuffers.pushBack(dstVertexBuffer);
    }

    hkArray<hkMeshMaterial*> dstMaterials;
    hkPointerMap<hkMeshMaterial*, hkMeshMaterial*> materialMap;

	const int numSections = sectionSet.getNumSections();

	hkLocalArray<hkMeshSectionCinfo> dstMeshSections(numSections);
	for (int i = 0; i < numSections; i++)
	{
		const hkMeshSection& srcSection = sectionSet.getSection(i);

        // Get/create a material compatible with the destination mesh system
		hkMeshMaterial* srcMaterial = srcSection.m_material;
        hkMeshMaterial* dstMaterial = materialMap.getWithDefault(srcMaterial, HK_NULL);

        if (!dstMaterial)
        {
            if (srcRegistry != dstRegistry)
            {
                // We need to create the material
                const char* name = HK_NULL;
				if( srcRegistry != HK_NULL )
				{
					name = srcRegistry->getMaterialName(srcSection.m_material);
				}
				else if( srcSection.m_material != HK_NULL )
				{
					name = srcSection.m_material->getName();
				}
					
                if (name)
                {
                    // Look it up
                    dstMaterial = dstRegistry->findMaterial(name);
                    if (dstMaterial)
                    {
                        dstMaterial->addReference();
                    }
                }

                if( !dstMaterial )
				{
					if( srcSection.m_material )
					{						
						dstMaterial = dstSystem->createMaterial();
						hkMeshMaterialUtil::convert(srcSection.m_material, dstMaterial);
					}
					else
					{
						dstMaterial = dstSystem->createMaterial();
					}
					
					if( name )
					{
						dstRegistry->registerMaterial(name, dstMaterial);
					}					
				}
            }
            else
            {
                dstMaterial = srcMaterial;
                dstMaterial->addReference();
            }

            dstMaterials.pushBack(dstMaterial);
            materialMap.insert(srcMaterial, dstMaterial);
        }

		hkMeshVertexBuffer* replaceBuffer = bufferMap.getWithDefault(srcSection.m_vertexBuffer, srcSection.m_vertexBuffer);

		hkMeshSectionCinfo& dstSection = dstMeshSections.expandOne();
		// Get the skinned vertex buffer
		dstSection.m_vertexBuffer = replaceBuffer;
        dstSection.m_material = dstMaterial;
		dstSection.m_primitiveType = srcSection.m_primitiveType;
		dstSection.m_numPrimitives = srcSection.m_numPrimitives;
		dstSection.m_indexType = srcSection.m_indexType;
		dstSection.m_indices = srcSection.m_indices;
		dstSection.m_vertexStartIndex = srcSection.m_vertexStartIndex;
		dstSection.m_transformIndex = srcSection.m_transformIndex;
    }

	hkMeshShape* dstMeshShape = dstSystem->createShape(dstMeshSections.begin(), dstMeshSections.getSize());

    hkReferencedObject::removeReferences(dstMaterials.begin(), dstMaterials.getSize());
    hkReferencedObject::removeReferences(dstVertexBuffers.begin(), dstVertexBuffers.getSize());

    return dstMeshShape;
}

/* static */hkMeshShape* hkMeshShapeUtil::replaceShapeVertexBuffers(hkMeshSystem* meshSystem, const hkMeshShape* meshShape, hkPointerMap<hkMeshVertexBuffer*, hkMeshVertexBuffer*>& bufferMap)
{
	hkMeshSectionLockSet sectionSet;
	sectionSet.addMeshSections(meshShape, hkMeshShape::ACCESS_VERTEX_BUFFER | hkMeshShape::ACCESS_INDICES);

	const int numSections = sectionSet.getNumSections();

	hkLocalArray<hkMeshSectionCinfo> dstMeshSections(numSections);
	for (int i = 0; i < numSections; i++)
	{
		const hkMeshSection& srcSection = sectionSet.getSection(i);

		hkMeshVertexBuffer* replaceBuffer = bufferMap.getWithDefault(srcSection.m_vertexBuffer, srcSection.m_vertexBuffer);

		hkMeshSectionCinfo& dstSection = dstMeshSections.expandOne();
		// Get the skinned vertex buffer
		dstSection.m_vertexBuffer = replaceBuffer;
		dstSection.m_material = srcSection.m_material;
		dstSection.m_primitiveType = srcSection.m_primitiveType;
		dstSection.m_numPrimitives = srcSection.m_numPrimitives;
		dstSection.m_indexType = srcSection.m_indexType;
		dstSection.m_indices = srcSection.m_indices;
		dstSection.m_vertexStartIndex = srcSection.m_vertexStartIndex;
		dstSection.m_transformIndex = srcSection.m_transformIndex;
    }

	hkMeshShape* dstMeshShape = meshSystem->createShape(dstMeshSections.begin(), dstMeshSections.getSize());
	return dstMeshShape;
}

/* static */hkResult hkMeshShapeUtil::transform(hkMeshShape* meshShape, const hkMatrix4& transformIn, hkBool normalize)
{
	hkMeshSectionLockSet sectionSet;
    sectionSet.addMeshSections(meshShape, hkMeshShape::ACCESS_VERTEX_BUFFER );

    // Get all the vertex buffers
    hkArray<hkMeshVertexBuffer*> srcVertexBuffers;
    sectionSet.findUniqueVertexBuffers(srcVertexBuffers);

    const int numBuffers = srcVertexBuffers.getSize();
    for (int i = 0; i < numBuffers; i++)
    {
        hkMeshVertexBuffer* buffer = srcVertexBuffers[i];

        // Transform them
		hkResult res = hkMeshVertexBufferUtil::transform(buffer, transformIn, normalize ? hkMeshVertexBufferUtil::TRANSFORM_NORMALIZE : 0);
        if (res != HK_SUCCESS)
        {
            return res;
        }
    }

    return HK_SUCCESS;
}

/* static */void hkMeshShapeUtil::calculateStatistics(const hkMeshShape* shape, Statistics& statsOut)
{
    calculateStatistics(&shape, 1, statsOut);
}

/* static */void hkMeshShapeUtil::calculateStatistics(const hkMeshShape** shapes, int numShapes, Statistics& statsOut)
{
    hkMeshSectionLockSet sectionSet;

    for (int i = 0; i < numShapes; i++)
    {
        const hkMeshShape* shape = shapes[i];
        sectionSet.addMeshSections(shape, hkMeshShape::ACCESS_INDICES | hkMeshShape::ACCESS_VERTEX_BUFFER);
    }

    statsOut.clear();
    //
    hkArray<hkMeshVertexBuffer*> buffers;
    sectionSet.findUniqueVertexBuffers(buffers);

    const int numBuffers = buffers.getSize();
    for (int i = 0; i < numBuffers; i++)
    {
        hkMeshVertexBuffer* buffer = buffers[i];
        statsOut.m_totalNumVertices += buffer->getNumVertices();
    }

    statsOut.m_numUniqueVertexBuffers = numBuffers;
	const int numSections = sectionSet.getNumSections();
	hkPointerMap<hkMeshMaterial*, int>::Temp materialMap; materialMap.reserve(numSections);
    for (int i = 0; i < numSections; i++)
    {
        const hkMeshSection& section = sectionSet.getSection(i);
        statsOut.m_totalNumPrimitives += section.m_numPrimitives;

        materialMap.insert(section.m_material, 1);
    }

    statsOut.m_numUniqueMaterials = materialMap.getSize();
}
			
/* static */void HK_CALL hkMeshShapeUtil::calcAabb(const hkMeshShape* shape, hkAabb& aabbOut)
{
	hkMeshSectionLockSet sectionLockSet;
	sectionLockSet.addMeshSections(shape, hkMeshShape::ACCESS_INDICES | hkMeshShape::ACCESS_VERTEX_BUFFER);
	hkArray<hkVector4> positions;
	hkArray<hkUint32> triIndices;

	aabbOut.setEmpty();

	for (int i = 0; i < sectionLockSet.getNumSections(); i++)
	{
		const hkMeshSection& section = sectionLockSet.getSection(i);

		// Get the positions
		hkMeshVertexBufferUtil::getElementVectorArray(section.m_vertexBuffer, hkVertexFormat::USAGE_POSITION, 0, positions);

		// 
		triIndices.clear();
		hkMeshPrimitiveUtil::appendTriangleIndices(section, triIndices);

		const int numIndices = triIndices.getSize();
		for (int j = 0; j < numIndices; j++)
		{
			aabbOut.includePoint(positions[triIndices[j]]);
		}
	}
}

//
//	Utility functions

namespace hkMeshShapeUtilImpl
{
	//
	//	The index buffer of a limited mesh section

	struct SectionIndexBuffer
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY, hkMeshShapeUtilImpl::SectionIndexBuffer);

		hkArray<hkUint8> m_indexData;		///< The new index data
		hkArray<int> m_vertexRemap;			///< An array of vertex indices. Gives the vertex index in this index buffer corresponding to an original vertex index in the original vertex buffer.
		hkBitField m_enabledBones;			///< A bit-field of bones acting on this section
	};

	//
	//	The vertex buffer of a limited mesh section

	struct SectionVertexBuffer
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY, hkMeshShapeUtilImpl::SectionVertexBuffer);

		hkMeshVertexBuffer* m_srcVb;			///< The original vertex buffer
		hkRefPtr<hkMeshVertexBuffer> m_dstVb;	///< The newly created vertex buffer
		hkBitField m_usedVerts;					///< A bit-field indicating whether a vertex is already in use by a section
		hkArray<int> m_verts;					///< An array of vertex indices in the original vertex buffer that need to be added to the new vertex buffer
	};

	//
	//	Returns the vertex index to be used by the given section instead of the given original index (used when duplicating vertices spanning across multiple sections)

	static HK_FORCE_INLINE int HK_CALL findVertexIndex(SectionVertexBuffer& sectionVb, SectionIndexBuffer& sectionIb, int origVtxIdx)
	{
		// Search the vertex in the section remap first
		if ( sectionIb.m_vertexRemap[origVtxIdx] >= 0 )
		{
			return sectionIb.m_vertexRemap[origVtxIdx];	// This vertex was already used by this section
		}

		// First time this section references the vertex. See if already in use by a different section
		int newVtxIdx;
		if ( sectionVb.m_usedVerts.get(origVtxIdx) )
		{
			// Vertex is in use, we need to create a new one
			newVtxIdx = sectionVb.m_verts.getSize();
			sectionVb.m_verts.pushBack(origVtxIdx);
		}
		else
		{
			// First time the vertex is used
			newVtxIdx = origVtxIdx;
		}

		// Mark as used and add to section remap
		sectionVb.m_usedVerts.set(origVtxIdx);
		sectionIb.m_vertexRemap[origVtxIdx] = newVtxIdx;
		return newVtxIdx;
	}

	//
	//	Computes a set of mesh sections that use at most maxBones. Returns true if any new sections had to be created

	static bool HK_CALL limitMeshSectionBones(	const hkMeshSection& srcSection, 
												SectionVertexBuffer& sectionVb,
												hkArray<hkMeshSectionCinfo>& limitedSectionsOut, hkArray<SectionIndexBuffer>& limitedIndexBuffersOut,
												int maxBonesPerSection, int numTotalMeshBones)
	{
		typedef hkMeshPrimitiveUtil::PrimitiveProvider PrimitiveProvider;

		// Get skinning data
		hkMeshVertexBuffer* srcVb			= sectionVb.m_srcVb;
		hkVertexFormat vfmt;				srcVb->getVertexFormat(vfmt);
		const int boneIndicesFmtIdx			= vfmt.findElementIndex(hkVertexFormat::USAGE_BLEND_MATRIX_INDEX, 0);
		const int boneWeightsFmtIdx			= hkMath::max2(vfmt.findElementIndex(hkVertexFormat::USAGE_BLEND_WEIGHTS, 0), vfmt.findElementIndex(hkVertexFormat::USAGE_BLEND_WEIGHTS_LAST_IMPLIED, 0));
		const int numInitialLimitedSections = limitedSectionsOut.getSize();
		bool maxIndexOverflow				= false;

		// Lock the vb
		hkMeshVertexBuffer::LockInput lockInput;
		hkMeshVertexBuffer::LockedVertices lockedVerts;
		srcVb->lock(lockInput, lockedVerts);

		// Get skinning data if any
		const int numVerts = srcVb->getNumVertices();
		hkArray<int> ptrBI;	
		hkArray<hkFloat32> ptrBW;
		if ( boneIndicesFmtIdx >= 0 )
		{
			ptrBI.setSize(numVerts << 2);	srcVb->getElementIntArray(lockedVerts, boneIndicesFmtIdx, ptrBI.begin());
			ptrBW.setSize(numVerts << 2);	srcVb->getElementVectorArray(lockedVerts, boneWeightsFmtIdx, ptrBW.begin());
		}

		// Iterate through the primitives and split them into groups
		hkBitField primitiveBones	(numTotalMeshBones, hkBitFieldValue::ZERO);
		hkBitField sectionBones		(numTotalMeshBones, hkBitFieldValue::ZERO);
		PrimitiveProvider			pReader(srcSection.m_indices, srcSection.m_primitiveType, srcSection.m_indexType);
		for (int pi = 0; pi < srcSection.m_numPrimitives; pi++)
		{
			pReader.readPrimitive();

			// Compute the index of the new section that is going to receive this primitive
			int bestSectionIdx = -1;
			if ( boneIndicesFmtIdx >= 0 )
			{
				// Load indices for the 3 verts
				hkIntVector biA;	biA.load<4>((const hkUint32*)&ptrBI[pReader.m_a << 2]);
				hkIntVector biB;	biB.load<4>((const hkUint32*)&ptrBI[pReader.m_b << 2]);
				hkIntVector biC;	biC.load<4>((const hkUint32*)&ptrBI[pReader.m_c << 2]);

				// Load weights for the 3 verts
				hkVector4 bwA;		bwA.load<4, HK_IO_NATIVE_ALIGNED>(&ptrBW[pReader.m_a << 2]);
				hkVector4 bwB;		bwB.load<4, HK_IO_NATIVE_ALIGNED>(&ptrBW[pReader.m_b << 2]);
				hkVector4 bwC;		bwC.load<4, HK_IO_NATIVE_ALIGNED>(&ptrBW[pReader.m_c << 2]);

				// Set the bones as used
				primitiveBones.assignAll(0);
				for (int k = 3; k >= 0; k--)
				{
					if ( bwA.getComponent<0>().isGreater(hkSimdReal_Eps) )	{	primitiveBones.set(biA.getComponent<0>());	}
					if ( bwB.getComponent<0>().isGreater(hkSimdReal_Eps) )	{	primitiveBones.set(biB.getComponent<0>());	}
					if ( bwC.getComponent<0>().isGreater(hkSimdReal_Eps) )	{	primitiveBones.set(biC.getComponent<0>());	}

					// Rotate all
					bwA.setPermutation<hkVectorPermutation::YZWX>(bwA);
					bwB.setPermutation<hkVectorPermutation::YZWX>(bwB);
					bwC.setPermutation<hkVectorPermutation::YZWX>(bwC);

					biA.setPermutation<hkVectorPermutation::YZWX>(biA);
					biB.setPermutation<hkVectorPermutation::YZWX>(biB);
					biC.setPermutation<hkVectorPermutation::YZWX>(biC);
				}

				// Try to add the primitive to an existing mesh section. Compute the fitting score for each section, i.e. the number of bones that need to be added
				int bestMergeScore = 0x7FFFFFFF;
				for (int k = limitedSectionsOut.getSize() - 1; k >= numInitialLimitedSections; k--)
				{
					const SectionIndexBuffer& newIb = limitedIndexBuffersOut[k];

					sectionBones.setOr(newIb.m_enabledBones, primitiveBones);
					const int initialNumBones	= newIb.m_enabledBones.bitCount();
					const int finalNumBones		= sectionBones.bitCount();
					const int deltaNumBones		= finalNumBones - initialNumBones;
					const int score				= (finalNumBones > maxBonesPerSection) ? 0x7FFFFFFF : deltaNumBones;

					if ( score < bestMergeScore )
					{
						bestMergeScore	= score;
						bestSectionIdx	= k;
					}
				}
			}
			else
			{
				// Not a skinned mesh section, simply return the first newSection available
				bestSectionIdx = limitedSectionsOut.getSize() - 1;
			}

			// See if we must add a new section
			if ( bestSectionIdx < 0 )
			{
				bestSectionIdx	= limitedSectionsOut.getSize();
			
				// Initialize the section
				hkMeshSectionCinfo& newSection	= limitedSectionsOut.expandOne();
				newSection.m_vertexBuffer		= srcVb;
				newSection.m_material			= srcSection.m_material;
				newSection.m_primitiveType		= hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST;	// No matter what, we save triangle lists
				newSection.m_numPrimitives		= 0;
				newSection.m_indexType			= (numVerts > 0xFFFF) ? hkMeshSection::INDEX_TYPE_UINT32 : hkMeshSection::INDEX_TYPE_UINT16;
				newSection.m_indices			= HK_NULL;
				newSection.m_vertexStartIndex	= -1;	// Should be ignored, as we specify an index buffer
				newSection.m_transformIndex		= -1;

				SectionIndexBuffer& newIb	= limitedIndexBuffersOut.expandOne();
				newIb.m_indexData.			setSize(0);
				newIb.m_enabledBones.		setSizeAndFill(0, numTotalMeshBones, 0);
				newIb.m_vertexRemap.		setSize(numVerts, -1);
			}

			// We found a section where there's still room, add there
			{
				hkMeshSectionCinfo& newSection	= limitedSectionsOut[bestSectionIdx];
				SectionIndexBuffer& newIb		= limitedIndexBuffersOut[bestSectionIdx];

				newIb.m_enabledBones.orWith(primitiveBones);
				const int growBy	= 3 * ((newSection.m_indexType == hkMeshSection::INDEX_TYPE_UINT32) ? sizeof(hkUint32) : sizeof(hkUint16));
				hkUint8* ptr		= newIb.m_indexData.expandBy(growBy);

				maxIndexOverflow	= ( newIb.m_enabledBones.findLastSet() >= maxBonesPerSection );
				PrimitiveProvider pWriter(ptr, newSection.m_primitiveType, newSection.m_indexType);

				const int vidxA = findVertexIndex(sectionVb, newIb, pReader.m_a);
				const int vidxB = findVertexIndex(sectionVb, newIb, pReader.m_b);
				const int vidxC = findVertexIndex(sectionVb, newIb, pReader.m_c);
				pWriter.writePrimitive(vidxA, vidxB, vidxC);
				newSection.m_numPrimitives++;
			}
		}

		// Unlock the vb
		srcVb->unlock(lockedVerts);

		// If there's only one limited sections, we didn't have to do anything
		return ( ((limitedSectionsOut.getSize() - numInitialLimitedSections) > 1) || maxIndexOverflow );
	}

	//
	//	Returns the index of the SectionVertexBuffer corresponding to the given hkMeshVertexBuffer

	static int HK_CALL getSectionVbFromSourceVb(const hkMeshVertexBuffer* srcVb, const hkArray<SectionVertexBuffer>& newVbs)
	{
		for (int vbIdx = newVbs.getSize() - 1; vbIdx >= 0; vbIdx--)
		{
			if ( newVbs[vbIdx].m_srcVb == srcVb )
			{
				return vbIdx;
			}
		}

		return -1;
	}

	static int HK_CALL getSectionVbFromDestinationVb(const hkMeshVertexBuffer* dstVb, const hkArray<SectionVertexBuffer>& newVbs)
	{
		for (int vbIdx = newVbs.getSize() - 1; vbIdx >= 0; vbIdx--)
		{
			if ( newVbs[vbIdx].m_dstVb == dstVb )
			{
				return vbIdx;
			}
		}

		return -1;
	}

	//
	//	Creates the new vertex buffer from the old one

	static void HK_CALL createNewVb(hkMeshSystem* meshSystem, SectionVertexBuffer& sectionVb)
	{
		hkMeshVertexBuffer* srcVb	= sectionVb.m_srcVb;
		const int numNewVerts		= sectionVb.m_verts.getSize();
		hkMeshVertexBuffer* dstVb	= meshSystem->createVertexBuffer(srcVb, numNewVerts);

		// Copy the vertices from source to destination
		hkMeshVertexBuffer::LockInput srcLockInput, dstLockInput;
		hkMeshVertexBuffer::LockedVertices srcVerts, dstVerts;

		srcVb->lock(srcLockInput, srcVerts);
		dstVb->lock(dstLockInput, dstVerts);

		// Copy each buffer
		for (int bi = 0; bi < srcVerts.m_numBuffers; bi++)
		{
			hkMeshVertexBuffer::LockedVertices::Buffer srcBuffer = srcVerts.m_buffers[bi];
			hkMeshVertexBuffer::LockedVertices::Buffer dstBuffer = dstVerts.m_buffers[bi];
			void* srcBasePtr = srcBuffer.m_start;
			void* dstBasePtr = dstBuffer.m_start;			

			for (int vi = 0; vi < numNewVerts; vi++)
			{
				const int oldVi		= sectionVb.m_verts[vi];
				srcBuffer.m_start	= hkAddByteOffset(srcBasePtr, srcBuffer.m_stride * oldVi);
				dstBuffer.m_start	= hkAddByteOffset(dstBasePtr, dstBuffer.m_stride * vi);

				hkMeshVertexBufferUtil::copy(srcBuffer, dstBuffer, 1);
			}
		}

		dstVb->unlock(dstVerts);
		srcVb->unlock(srcVerts);

		sectionVb.m_dstVb .setAndDontIncrementRefCount(dstVb);
	}

	//
	//	Initializes a section vertex buffer

	static void HK_CALL initVb(SectionVertexBuffer& sectionVb, hkMeshVertexBuffer* srcVb)
	{
		const int numVerts	= srcVb->getNumVertices();

		sectionVb.m_srcVb		= srcVb;
		sectionVb.m_dstVb		= HK_NULL;
		sectionVb.m_usedVerts.	setSizeAndFill(0, numVerts, 0);
		sectionVb.m_verts.		setSize(numVerts);

		for (int k = numVerts - 1; k >= 0; k--)
		{
			sectionVb.m_verts[k] = k;
		}
	}

	//
	//	Remaps the bone indices used by the given section

	static void HK_CALL remapBoneIndices(SectionVertexBuffer& sectionVb, SectionIndexBuffer& sectionIb, const hkArray<hkInt16>& newFromOldBoneIndex)
	{
		// See if there's any skinning info
		hkMeshVertexBuffer* dstVb		= sectionVb.m_dstVb;
		hkVertexFormat vfmt;			dstVb->getVertexFormat(vfmt);	
		const int boneIndicesFmtIdx		= vfmt.findElementIndex(hkVertexFormat::USAGE_BLEND_MATRIX_INDEX, 0);
		if ( boneIndicesFmtIdx < 0 )
		{
			return;	// Nothing to do, buffer not skinned!
		}

		// Lock the buffer
		hkMeshVertexBuffer::LockInput lockInput;
		hkMeshVertexBuffer::LockedVertices lockedVerts;
		dstVb->lock(lockInput, lockedVerts);

		// Get the bone indices
		const int numVerts	= dstVb->getNumVertices();
		hkArray<int> ptrBI;	ptrBI.setSize(numVerts << 2);
		dstVb->getElementIntArray(lockedVerts, boneIndicesFmtIdx, ptrBI.begin());

		// Remap all vertices used by this section
		for (int origVtxIdx = sectionIb.m_vertexRemap.getSize() - 1; origVtxIdx >= 0; origVtxIdx--)
		{
			const int newVtxIdx = sectionIb.m_vertexRemap[origVtxIdx];
			if ( newVtxIdx < 0 )
			{
				continue;	// Vertex not in use, ignore!
			}

			int* boneIdx = &ptrBI[newVtxIdx << 2];
			boneIdx[0] = newFromOldBoneIndex[boneIdx[0]];
			boneIdx[1] = newFromOldBoneIndex[boneIdx[1]];
			boneIdx[2] = newFromOldBoneIndex[boneIdx[2]];
			boneIdx[3] = newFromOldBoneIndex[boneIdx[3]];
		}

		// Write back remapped indices and unlock
		dstVb->setElementIntArray(lockedVerts, boneIndicesFmtIdx, ptrBI.begin());
		dstVb->unlock(lockedVerts);
	}
}

//
//	Returns a mesh shape with at most maxBones in each mesh section

const hkMeshShape* HK_CALL hkMeshShapeUtil::createMeshWithLimitedNumBones(hkMeshSystem* meshSystem, const hkMeshShape* srcShape, int maxBonesPerSection, int numTotalMeshBones, hkArray<hkMeshBoneIndexMapping>& boneMappingsOut)
{
	typedef hkMeshShapeUtilImpl::SectionVertexBuffer	SectionVertexBuffer;
	typedef hkMeshShapeUtilImpl::SectionIndexBuffer		SectionIndexBuffer;

	hkArray<hkMeshSectionCinfo> newSections;
	hkArray<SectionVertexBuffer> newVbs;
	hkArray<SectionIndexBuffer> newIbs;

	// Limit the number of bones in each section
	bool needNewMesh = false;
	const int numSrcSections = srcShape->getNumSections();
	for (int si = 0; si < numSrcSections; si++)
	{
		// Lock the section and get the vertex buffer
		hkMeshSection section;			srcShape->lockSection(si, hkMeshShape::ACCESS_VERTEX_BUFFER | hkMeshShape::ACCESS_INDICES, section);
		hkMeshVertexBuffer* sectionVb	= section.m_vertexBuffer;

		// Locate the vertex buffer, or add new if not already there
		int vbIdx = hkMeshShapeUtilImpl::getSectionVbFromSourceVb(sectionVb, newVbs);
		if ( vbIdx < 0 )
		{
			// We need a new vertex buffer
			vbIdx = newVbs.getSize();
			hkMeshShapeUtilImpl::initVb(newVbs.expandOne(), sectionVb);
		}

		// Compute the limited mesh sections
		const bool haveNewSections = hkMeshShapeUtilImpl::limitMeshSectionBones(section, newVbs[vbIdx], newSections, newIbs, maxBonesPerSection, numTotalMeshBones);
		
		// Decide whether we need to build a new mesh
		needNewMesh = needNewMesh || haveNewSections;
		srcShape->unlockSection(section);
	}

	// Return the limited mesh
	const hkMeshShape* retShape = srcShape;
	if ( needNewMesh )
	{
		// Create the new vertex buffers
		for (int k = newVbs.getSize() - 1; k >= 0; k--)
		{
			createNewVb(meshSystem, newVbs[k]);
		}

		// Assign them to the new mesh sections
		const int numNewSections = newSections.getSize();
		for (int si = 0; si < numNewSections; si++)
		{
			hkMeshSectionCinfo& newSection = newSections[si];
			const int vbIdx = hkMeshShapeUtilImpl::getSectionVbFromSourceVb(newSection.m_vertexBuffer, newVbs);
			newSections[si].m_vertexBuffer	= newVbs[vbIdx].m_dstVb;
			newSections[si].m_indices		= newIbs[si].m_indexData.begin();
		}

		// Create the new mesh
		retShape = meshSystem->createShape(newSections.begin(), numNewSections);

		// We've created a new shape, we need to re-index the bones in each section to match our limits
		hkMeshBoneIndexMapping* boneMappings	= boneMappingsOut.expandBy(numNewSections);
		hkArray<hkInt16> revMapping;			revMapping.setSize(numTotalMeshBones);

		for (int si = 0; si < numNewSections; si++)
		{
			hkMeshBoneIndexMapping& boneMapping = boneMappings[si];
			const hkBitField& sectionBones		= newIbs[si].m_enabledBones;
			const int numUsedSectionBones		= sectionBones.bitCount();
			boneMapping.m_mapping.reserve(numUsedSectionBones);

			// Compute mappings
			revMapping.setSize(0);
			revMapping.setSize(numTotalMeshBones, -1);
			for (int oldIdx = 0; oldIdx < numTotalMeshBones; oldIdx++)
			{
				if ( sectionBones.get(oldIdx) )
				{
					revMapping[oldIdx] = (hkInt16)boneMapping.m_mapping.getSize();
					boneMapping.m_mapping.pushBack((hkInt16)oldIdx);
				}
			}

			// Remap the vertices
			hkMeshSectionCinfo& newSection = newSections[si];
			const int vbIdx = hkMeshShapeUtilImpl::getSectionVbFromDestinationVb(newSection.m_vertexBuffer, newVbs);
			remapBoneIndices(newVbs[vbIdx], newIbs[si], revMapping);
		}
	}
	
	// Return either the new or original unchanged mesh
	return retShape;
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
