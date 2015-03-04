/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/hkMeshVertexBuffer.h>
#include <Common/GeometryUtilities/Mesh/hkMeshShape.h>
#include <Common/GeometryUtilities/Mesh/Skin/hkSkinnedMeshShape.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Container/BitField/hkBitField.h>

//
//	Constructor

hkSkinnedMeshShape::hkSkinnedMeshShape()
:	hkReferencedObject()
{}

//
//	Serialization constructor

hkSkinnedMeshShape::hkSkinnedMeshShape(class hkFinishLoadedObjectFlag flag)
:	hkReferencedObject(flag)
{}

//
//	Destructor

hkSkinnedMeshShape::~hkSkinnedMeshShape()
{}

//
//	Constructor

hkSkinnedMeshShape::BoneSection::BoneSection()
:	m_meshBuffer(HK_NULL)
,	m_startBoneSetId(BoneSetId::invalid())
,	m_numBoneSets(0)
{}

//
//	Serialization constructor

hkSkinnedMeshShape::BoneSection::BoneSection(class hkFinishLoadedObjectFlag flag)
:	m_meshBuffer(flag)
{}

//
//	Prints a lot of debug info about the mesh

void hkSkinnedMeshShape::dbgOut() const
{
	hkStringBuf strb;
	const int numMeshes = getNumBoneSections();
	const int numParts	= getNumParts();

	HK_REPORT("---------------------------------------------------");
	strb.printf("Num bone sections: %d. Num parts: %d.", numMeshes, numParts);
	HK_REPORT(strb);
	for (int mi = 0; mi < numMeshes; mi++)
	{
		BoneSection mesh;
		getBoneSection(mi, mesh);

		hkMeshShape* dynMesh = mesh.m_meshBuffer;
 		hkArray<hkVector4> positions;

		int numSubMeshes = dynMesh->getNumSections();
		strb.printf("Bone section %d. Start boneSet %d, numBoneSets %d. Num sections %d", mi, mesh.m_startBoneSetId.value(), mesh.m_numBoneSets, numSubMeshes);
		HK_REPORT(strb);

		// Print all parts in this mesh
		for (int pi = 0; pi < numParts; pi++)
		{
			Part part;
			getPart(pi, part);

			if ( (part.m_boneSetId >= mesh.m_startBoneSetId) && (part.m_boneSetId.value() < mesh.m_startBoneSetId.value() + mesh.m_numBoneSets) )
			{
				hkMeshSection subMesh;
				dynMesh->lockSection(part.m_meshSectionIndex, hkMeshShape::ACCESS_VERTEX_BUFFER | hkMeshShape::ACCESS_INDICES, subMesh);

				strb.printf("Part %d. BoneSet %d. Sub-mesh %d. Start vertex %d. NumVerts %d. Start Index %d. NumIndices %d.",
					pi, part.m_boneSetId.value(), part.m_meshSectionIndex, part.m_startVertex, part.m_numVertices, part.m_startIndex, part.m_numIndices);
				HK_REPORT(strb);

				// Print vertices
				{
					hkMeshVertexBuffer* vb = subMesh.m_vertexBuffer;
					hkMeshVertexBuffer::LockedVertices lockedVerts;
					hkMeshVertexBuffer::LockInput lockInput;
					lockInput.m_startVertex = part.m_startVertex;
					lockInput.m_numVertices = part.m_numVertices;
					lockInput.m_lockFlags	= hkMeshVertexBuffer::ACCESS_READ;
					vb->lock(lockInput, lockedVerts);
					{
						hkVertexFormat vfmt;
						vb->getVertexFormat(vfmt);

						const int positionsIdx = vfmt.findElementIndex(hkVertexFormat::USAGE_POSITION, 0);
						positions.setSize(part.m_numVertices);
						hkArray<hkFloat32>::Temp va; va.setSize(4*part.m_numVertices);
						vb->getElementVectorArray(lockedVerts, positionsIdx, va.begin());
						for (int i=0; i<part.m_numVertices; ++i)
						{
							positions[i].load<4,HK_IO_NATIVE_ALIGNED>(&va[4*i]);
						}

						for (int vi = 0; vi < part.m_numVertices; vi++)
						{
							const hkVector4 v = positions[vi];
							strb.printf("Vtx %d.	(%f, %f, %f)", vi + part.m_startVertex, v(0), v(1), v(2));
							HK_REPORT(strb);
						}
					}
					vb->unlock(lockedVerts);
				}

				// Print indices
				{
					HK_ASSERT(0x3e2cc0f0, (part.m_numIndices % 3) == 0);
					if ( subMesh.m_indexType == hkMeshSection::INDEX_TYPE_UINT16 )
					{
						hkInt16* ibPtr = (hkInt16*)subMesh.m_indices;
						for (int ii = 0; ii < part.m_numIndices; ii += 3)
						{
							const hkInt16 idxa = ibPtr[part.m_startIndex + ii + 0];
							const hkInt16 idxb = ibPtr[part.m_startIndex + ii + 1];
							const hkInt16 idxc = ibPtr[part.m_startIndex + ii + 2];
							strb.printf("Tri (%d, %d, %d).", idxa, idxb, idxc);
							HK_REPORT(strb);
						}
					}
					else if ( subMesh.m_indexType == hkMeshSection::INDEX_TYPE_UINT32 )
					{
						hkUint32* ibPtr = (hkUint32*)subMesh.m_indices;
						for (int ii = 0; ii < part.m_numIndices; ii += 3)
						{
							const hkUint32 idxa = ibPtr[part.m_startIndex + ii + 0];
							const hkUint32 idxb = ibPtr[part.m_startIndex + ii + 1];
							const hkUint32 idxc = ibPtr[part.m_startIndex + ii + 2];
							strb.printf("Tri (%d, %d, %d).", idxa, idxb, idxc);
							HK_REPORT(strb);
						}
					}
				}

				dynMesh->unlockSection(subMesh);
			}
		}
	}
}

//
//	Computes the bone mapping for the given bone section

void hkSkinnedMeshShape::computeBoneSectionMapping(int boneSectionIdx, hkMeshBoneIndexMapping& mappingOut) const
{
	// Get section
	BoneSection section;
	getBoneSection(boneSectionIdx, section);

	// Locate the min and max used bones
	hkInt16 minUsedBoneIdx = 0x7FFF;
	hkInt16 maxUsedBoneIdx = 0;
	hkArray<hkInt16> bones;
	for (hkInt16 k = 0; k < section.m_numBoneSets; k++)
	{
		// Get bone-set
		const BoneSetId boneSetId(k + section.m_startBoneSetId.value());
		getBoneSetBones(boneSetId, bones);

		// Update min and max bones
		for (int bi = bones.getSize() - 1; bi >= 0; bi--)
		{
			minUsedBoneIdx = hkMath::min2(minUsedBoneIdx, bones[bi]);
			maxUsedBoneIdx = hkMath::max2(maxUsedBoneIdx, bones[bi]);
		}
	}

	// Alloc a bit-field and gather only the used bones
	const int maxNumBones		= maxUsedBoneIdx - minUsedBoneIdx + 1;
	hkBitField usedBones(maxNumBones, hkBitFieldValue::ZERO);
	for (hkInt16 k = 0; k < section.m_numBoneSets; k++)
	{
		// Get bone-set
		const BoneSetId boneSetId(k + section.m_startBoneSetId.value());
		getBoneSetBones(boneSetId, bones);

		// Update min and max bones
		for (int bi = bones.getSize() - 1; bi >= 0; bi--)
		{
			usedBones.set(bones[bi] - minUsedBoneIdx);
		}
	}

	// Count the number of actually used bones
	for (int k = 0; k < usedBones.getSize(); k++)
	{
		if ( usedBones.get(k) )
		{
			mappingOut.m_mapping.pushBack((hkInt16)(minUsedBoneIdx + k));
		}
	}
}

//
//	Computes all mesh section mappings. There will be one mapping per bone section, stored in boneSectionMappingsOut. For each unique mesh section
//	in each bone section, there will be an entry in the perSectionBoneMappingIdicesOut, indicating which of the boneSectionMappingsOut to use.

void hkSkinnedMeshShape::computeMeshSectionMappings(hkArray<hkMeshBoneIndexMapping>& boneSectionMappingsOut, hkArray<hkUint16>& perSectionBoneMappingIndicesOut)
{
	// Compute bone section mappings first. Collect mesh shapes from all bone sections in the process
	const int numBoneSections = getNumBoneSections();
	boneSectionMappingsOut.setSize(numBoneSections);
	hkArray<const hkMeshShape*> allMeshShapes;	allMeshShapes.setSize(numBoneSections);
	hkArray<int> meshSectionBaseIdx;			meshSectionBaseIdx.setSize(numBoneSections);

	int numTotalMeshSections = 0;
	for (int bi = 0; bi < numBoneSections; bi++)
	{
		computeBoneSectionMapping(bi, boneSectionMappingsOut[bi]);

		BoneSection boneSection;	getBoneSection(bi, boneSection);
		const hkMeshShape* mesh		= boneSection.m_meshBuffer;
		allMeshShapes[bi]			= mesh;

		// Check if we already have this mesh in our list
		int k = bi - 1;
		for (; k >= 0; k--)
		{
			if ( allMeshShapes[k] == mesh )
			{
				break;
			}
		}
		if ( k  < 0 )
		{
			// This is the first time we've seen the mesh, add all its mesh sections
			meshSectionBaseIdx[bi] = numTotalMeshSections;
			numTotalMeshSections += mesh->getNumSections();
		}
		else
		{
			// We've seen this mesh before, use the same base mesh section index
			meshSectionBaseIdx[bi] = meshSectionBaseIdx[k];
		}
	}

	// For each part, check its bone-set and mesh section, we'll compute the index based on them
	perSectionBoneMappingIndicesOut.setSize(numTotalMeshSections, 0xFFFF);
	const int numParts = getNumParts();
	for (int pi = numParts - 1; pi >= 0; pi--)
	{
		// Get part and its bone-set
		Part part;					getPart(pi, part);
		const BoneSetId boneSetId	= part.m_boneSetId;

		// Locate the bone section from the bone-set
		int bi = numBoneSections - 1;
		for (; bi >= 0; bi--)
		{
			BoneSection boneSection;	getBoneSection(bi, boneSection);
			const hkInt16 relBoneSetId	= (hkInt16)(boneSetId.value() - boneSection.m_startBoneSetId.value());
			if ( (relBoneSetId >= 0) && (relBoneSetId < boneSection.m_numBoneSets))
			{
				break;
			}
		}
		HK_ASSERT(0x6af6b60, bi >= 0);

		// The part drives a mesh section in the bone section bi. All mesh sections in bone section bi share the same bone mapping
		const int globalMeshSectionIdx = meshSectionBaseIdx[bi] + part.m_meshSectionIndex;
		perSectionBoneMappingIndicesOut[globalMeshSectionIdx] = (hkUint16)bi;
	}
}

//
//	Constructor

hkStorageSkinnedMeshShape::hkStorageSkinnedMeshShape()
:	hkSkinnedMeshShape()
,	m_name(HK_NULL)
{}

//
//	Serialization constructor

hkStorageSkinnedMeshShape::hkStorageSkinnedMeshShape(class hkFinishLoadedObjectFlag flag)
:	hkSkinnedMeshShape(flag)
,	m_bonesBuffer(flag)
,	m_boneSets(flag)
,	m_boneSections(flag)
,	m_parts(flag)
,	m_name(flag)
{}

//
//	Gets the number of bone sections

int hkStorageSkinnedMeshShape::getNumBoneSections() const
{
	return m_boneSections.getSize();
}

//
//	Returns the bone section at the given index

void hkStorageSkinnedMeshShape::getBoneSection(int boneSectionIndex, BoneSection& boneSectionOut) const
{
	boneSectionOut = m_boneSections[boneSectionIndex];
}

//
//	Returns the bone set associated with the given id

void hkStorageSkinnedMeshShape::getBoneSet(BoneSetId boneSetId, BoneSet& boneSetOut) const
{
	boneSetOut = m_boneSets[boneSetId.value()];
}

//
//	Returns the bones associated with the given BoneSet

void hkStorageSkinnedMeshShape::getBoneSetBones(BoneSetId boneSetId, hkArray<hkInt16>& bonesOut) const
{
	const BoneSet& boneSet = m_boneSets[boneSetId.value()];

	bonesOut.setSize(0);
	bonesOut.append(&m_bonesBuffer[boneSet.m_boneBufferOffset], boneSet.m_numBones);
}

//
//	Returns the number of parts

int hkStorageSkinnedMeshShape::getNumParts() const
{
	return m_parts.getSize();
}

//
//	Returns the part at the given index

void hkStorageSkinnedMeshShape::getPart(int partIndex, Part& partOut) const
{
	partOut = m_parts[partIndex];
}

//
//	Sorts the parts by increasing bone indices

void hkStorageSkinnedMeshShape::sortParts()
{
	hkAlgorithm::quickSort(m_parts.begin(), m_parts.getSize(), Part::less);
}

//
//	Returns the class type

const hkClass* hkStorageSkinnedMeshShape::getClassType() const
{
	return &hkStorageSkinnedMeshShapeClass;
}

//
//	Adds a new bone section

void hkStorageSkinnedMeshShape::addBoneSection(hkMeshShape* meshShape, BoneSetId startBoneSetId, hkInt16 numBoneSets)
{
	// Allocate a new bone section
	BoneSection& m = m_boneSections.expandOne();

	m.m_meshBuffer		= meshShape;
	m.m_startBoneSetId	= startBoneSetId;
	m.m_numBoneSets		= numBoneSets;
}

//
//	Marks the start of a new piece

void hkStorageSkinnedMeshShape::addPart(const Part& p)
{	
	// Append part and update bone info
	m_parts.pushBack(p);
}

//
//	Gets the name of this shape

const char* hkStorageSkinnedMeshShape::getName() const
{
	return m_name;
}

//
//	Sets the name of this shape

void hkStorageSkinnedMeshShape::setName(const char* name)
{
	m_name = name;
}

//
//	Creates a BoneSet out of the provided array of bones. Returns the index of the newly created BoneSet

hkSkinnedMeshShape::BoneSetId hkStorageSkinnedMeshShape::addBoneSet(const hkInt16* HK_RESTRICT boneIndices, int numBones)
{
	// Add new bone set
	const BoneSetId boneSetId(m_boneSets.getSize());
	BoneSet& boneSet	= m_boneSets.expandOne();

	// Init bone set
	boneSet.m_boneBufferOffset	= (hkUint16)m_bonesBuffer.getSize();
	boneSet.m_numBones			= (hkUint16)numBones;
	
	// Add bones to the buffer and return
	m_bonesBuffer.append(boneIndices, numBones);
	return boneSetId;
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
