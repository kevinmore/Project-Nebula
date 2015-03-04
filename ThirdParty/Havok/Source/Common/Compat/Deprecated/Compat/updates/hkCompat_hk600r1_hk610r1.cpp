/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Compat/hkCompat.h>
#include <Common/Compat/Deprecated/Compat/hkCompatUtil.h>
#include <Common/Base/Config/hkConfigVersion.h>
#include <Common/Compat/Deprecated/Version/hkVersionRegistry.h>
#include <Common/Serialize/Version/hkVersionUtil.h>
#include <Common/Compat/Deprecated/Version/hkVersionUtilOld.h>
#include <Common/Serialize/Util/hkBuiltinTypeRegistry.h>
#include <Common/Base/hkBase.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

#include <Common/Compat/Deprecated/Compat/hkHavokAllClasses.h>

namespace hkCompat_hk600r1_hk610r1
{
	static hkBool classIsDerivedFrom(const hkClass* klass, const char* baseName)
	{
		while (klass && hkString::strCmp(klass->getName(), baseName) != 0)
		{
			klass = klass->getParent();
		}
		return klass != HK_NULL;
	}

	static void removeUnwantedObjects( hkArray<hkVariant>& objectsInOut, const char*const* typeNameList, hkObjectUpdateTracker& tracker )
	{
		hkArray<hkVariant> wantedObjects;
		wantedObjects.reserve(objectsInOut.getSize());
		for( int i = 0; i < objectsInOut.getSize(); ++i )
		{
			const hkClass* klass = objectsInOut[i].m_class;
			bool wanted = true;
			for( int k = 0; typeNameList[k] != HK_NULL; ++k )
			{
				if( classIsDerivedFrom(klass, typeNameList[k]) )
				{
					tracker.replaceObject(objectsInOut[i].m_object, HK_NULL, HK_NULL);
					wanted = false;
					break;
				}
			}
			if( wanted )
			{
				wantedObjects.pushBackUnchecked(objectsInOut[i]);
			}
		}
		objectsInOut.swap(wantedObjects);
	}

#define REMOVED(TYPE) { 0,0, hkVersionRegistry::VERSION_REMOVED, TYPE, HK_NULL }
#define BINARY_IDENTICAL(OLDSIG,NEWSIG,TYPE) { OLDSIG, NEWSIG, hkVersionRegistry::VERSION_MANUAL, TYPE, HK_NULL }

	static void Update_ignore( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
	}

	static void UpdateListShape( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkClassMemberAccessor oldMember(oldObj, "flags");
		hkClassMemberAccessor newMember(newObj, "flags");
		newMember.asUint16() = hkUint16(oldMember.asUint32());


		hkClassMemberAccessor newMember2(newObj, "numDisabledChildren");
		newMember2.asUint16() = 0;
	}

	static void Update_Assert( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		HK_ASSERT2(0xad808271, false, "Instances of this class should not be directly referenced. It's only expected to be member of another class.");
	}

	static void Update_Empty( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
	}

	static void hkdBreakableShape_Update( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// Iterate over the connections array & update the objects.
		//
		{
			hkClassMemberAccessor oldMember(oldObj, "findInitialContactPoints");
			hkClassMemberAccessor newMember(newObj, "flags");

			// Set flags according to FLAG_FIND_INITIAL_CONTACT_POINTS
			int enumValue; 
			hkClassAccessor newClassAccessor(newObj);
			newClassAccessor.getClass().getEnumByName("Flags")->getValueOfName("FLAG_FIND_INITIAL_CONTACT_POINTS", &enumValue);
			newMember.asInt8() = oldMember.asBool() ? hkInt8(enumValue) : hkInt8(0);
		}

		// Strength is renamed to contactArea in hkdBreakableShape::Connection struct. We must copy the value by manually.
		//
		{
			hkClassMemberAccessor oldMember(oldObj, "connections");
			
			hkClassMemberAccessor newMember(newObj, "connections");

			hkClassMemberAccessor::SimpleArray& oldConnections = oldMember.asSimpleArray();
			hkClassMemberAccessor::SimpleArray& newConnections = newMember.asSimpleArray();

			hkVariant oldConnectionVariant;
			hkVariant newConnectionVariant;

			oldConnectionVariant.m_class = &oldMember.object().getClass();
			newConnectionVariant.m_class = &newMember.object().getClass();

			const int oldStriding = oldMember.getClassMember().getStructClass().getObjectSize();
			const int newStriding = newMember.getClassMember().getStructClass().getObjectSize();

			for (int i = 0; i < newConnections.size; i++)			
			{
				oldConnectionVariant.m_object = static_cast<char*>( hkAddByteOffset(oldConnections.data, i * oldStriding) );
				newConnectionVariant.m_object = static_cast<char*>( hkAddByteOffset(newConnections.data, i * newStriding) );

				hkClassMemberAccessor oldStrength(oldConnectionVariant, "strength");
				hkClassMemberAccessor newContactArea(newConnectionVariant, "contactArea");

				// Copy the value
				newContactArea.asReal() = oldStrength.asReal();
			}

			// simple pointers changed to a RefPtr<> for m_physicsShape, m_geometry, m_graphicsShape, m_dynamicFracture


		}

	}

	// HCL-519 - SPU implementation of Move Particles requires pairs to be ordered by vertex Index

	// Move Particle Pairs entries as in 6.0.0
	struct hclMoveParticlesOperatorPair
	{
		hkUint32 m_vertexIndex;
		hkUint32 m_particleIndex;
	};

	static HK_FORCE_INLINE hkBool pairless( const hclMoveParticlesOperatorPair& a, const hclMoveParticlesOperatorPair& b )
	{
		return a.m_vertexIndex < b.m_vertexIndex;

	}

	static void Update_hclMoveParticlesOperator( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// HCL-519 - Entries must be ordered by reference index
		hkClassMemberAccessor oldMember (oldObj, "vertexParticlePairs");
		hkClassMemberAccessor newMember (newObj, "vertexParticlePairs");

		hkClassMemberAccessor::SimpleArray& arrayOld = oldMember.asSimpleArray();
		hkClassMemberAccessor::SimpleArray& arrayNew = newMember.asSimpleArray();

		arrayNew = arrayOld; // Reuse the same memory

		hclMoveParticlesOperatorPair* entries = reinterpret_cast<hclMoveParticlesOperatorPair*> (arrayNew.data);

		// HCL-533 - We need to reorder by "referenceVertex" in order to execute on SPU
		hkSort(entries, arrayNew.size, pairless );
	}

	// HCL-695 : Removal of redundant data in hclBlendSomeVerticesOperator, and
	// HCL-502 : Blend some vertices on SPU (requires A,B->A blends to be converted into B,A->A)

	// BlendSomeVertices entry as in 6.0.0
	struct hclBlendSomeVerticesOperatorEntry600
	{
		hkUint32 m_indexA;
		hkUint32 m_indexB;
		hkUint32 m_indexC;
		hkReal m_blendWeight;
	};

	// BlendSomeVertices entry as in 6.1.0
	struct hclBlendSomeVerticesOperatorEntry610
	{
		hkUint32 m_vertexIndex;
		hkReal m_blendWeight;
	};

	static void Update_hclBlendSomeVerticesOperator( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		const int bufA = hkClassMemberAccessor(oldObj, "bufferIdx_A").asUint32();
		const int bufB = hkClassMemberAccessor(oldObj, "bufferIdx_B").asUint32();
		const int bufC = hkClassMemberAccessor(oldObj, "bufferIdx_C").asUint32();

		// HCL-502 - Switch buffers if necessary (weight modified later on as well)
		const bool switchBuffers = (bufA==bufC);
		if (switchBuffers)
		{
			hkClassMemberAccessor(newObj, "bufferIdx_A").asUint32() = bufB;
			hkClassMemberAccessor(newObj, "bufferIdx_B").asUint32() = bufA;
		}

		hkClassMemberAccessor oldMember (oldObj, "vertexTriples");
		hkClassMemberAccessor newMember (newObj, "blendEntries");

		hkClassMemberAccessor::SimpleArray& arrayOld = oldMember.asSimpleArray();
		struct Array{ void* data; int size; int capacity; };
		Array& arrayNew = *(Array*)newMember.asRaw();

		const int numEntries = arrayOld.size;

		hclBlendSomeVerticesOperatorEntry610* newEntries = hkAllocateChunk<hclBlendSomeVerticesOperatorEntry610> (numEntries, HK_MEMORY_CLASS_CLOTH_DATA);
		const hclBlendSomeVerticesOperatorEntry600* oldEntries = reinterpret_cast<hclBlendSomeVerticesOperatorEntry600*> (arrayOld.data);

		arrayNew.data = newEntries;
		arrayNew.size = numEntries;
		arrayNew.capacity = numEntries;

		for (int i=0; i<numEntries; ++i)
		{
			newEntries[i].m_vertexIndex = oldEntries[i].m_indexA;
			newEntries[i].m_blendWeight = switchBuffers ? (1.0f-oldEntries[i].m_blendWeight) : oldEntries[i].m_blendWeight;
		}		
	}

	// HCL-704 : We now store a totalMass in simulationCloth data

	// Particle data in 6.0
	struct hclSimClothDataParticleData_600
	{
		hkReal m_mass;
		hkReal m_invMass;
		hkReal m_radius;
		hkReal m_friction;
	};

	static void Update_hclSimClothData ( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkClassMemberAccessor::SimpleArray particleDataArray = hkClassMemberAccessor(newObj, "particleDatas").asSimpleArray();

		const int numParticles = particleDataArray.size;

		hclSimClothDataParticleData_600* particleDatas = reinterpret_cast<hclSimClothDataParticleData_600*> (particleDataArray.data);

		hkReal totalMass = 0.0f;
		for (int p=0; p<numParticles; ++p)
		{
			totalMass += particleDatas[p].m_mass;;
		}

		hkClassMemberAccessor(newObj, "totalMass").asReal() = totalMass;
	}

static const hkVersionRegistry::ClassAction s_updateActions[] =
{
	// variants
	{ 0x6728e4b7, 0x6728e4b7, hkVersionRegistry::VERSION_VARIANT, "hkaBoneAttachment", HK_NULL },
	{ 0xf598a34e, 0xf598a34e, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainer", HK_NULL },
	{ 0x853a899c, 0x853a899c, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainerNamedVariant", HK_NULL },
	{ 0x3d4223b3, 0xdce3ca6b, hkVersionRegistry::VERSION_VARIANT | hkVersionRegistry::VERSION_COPY, "hkMemoryResourceHandle", HK_NULL },
	{ 0xbe6765dd, 0xbe6765dd, hkVersionRegistry::VERSION_VARIANT, "hkxMaterial", HK_NULL },
	{ 0x914da6c1, 0x914da6c1, hkVersionRegistry::VERSION_VARIANT, "hkxAttribute", HK_NULL },
	{ 0x1667c01c, 0x1667c01c, hkVersionRegistry::VERSION_VARIANT, "hkxAttributeGroup", HK_NULL },
	{ 0x06af1b5a, 0x06af1b5a, hkVersionRegistry::VERSION_VARIANT, "hkxNode", HK_NULL },
	{ 0x72e8e849, 0x72e8e849, hkVersionRegistry::VERSION_VARIANT, "hkxMesh", HK_NULL },
	{ 0x912c8863, 0x912c8863, hkVersionRegistry::VERSION_VARIANT, "hkxMeshSection", HK_NULL },
	{ 0x64e9a03c, 0x64e9a03c, hkVersionRegistry::VERSION_VARIANT, "hkxMeshUserChannelInfo", HK_NULL },
	{ 0x445a443a, 0x445a443a, hkVersionRegistry::VERSION_VARIANT, "hkxAttributeHolder", HK_NULL },
	{ 0x1388d601, 0x1388d601, hkVersionRegistry::VERSION_VARIANT, "hkCustomAttributesAttribute", HK_NULL },
	{ 0xbff19005, 0xbff19005, hkVersionRegistry::VERSION_VARIANT, "hkCustomAttributes", HK_NULL },
	{ 0x02ea23f0, 0x02ea23f0, hkVersionRegistry::VERSION_VARIANT, "hkxMaterialTextureStage", HK_NULL },
	{ 0x8e9b1727, 0x8e9b1727, hkVersionRegistry::VERSION_HOMOGENEOUSARRAY, "hkxVertexBuffer", HK_NULL },

	{ 0xec536340, 0x93561016, hkVersionRegistry::VERSION_COPY, "hkMemoryResourceContainer", HK_NULL },
	{ 0xda8c7d7d, 0x04e94146, hkVersionRegistry::VERSION_COPY, "hkResourceHandle", HK_NULL },
	{ 0xda8c7d7d, 0x04e94146, hkVersionRegistry::VERSION_COPY, "hkResourceContainer", HK_NULL },

	// Added a name and orig paths to textures
	{ 0xf64b134c, 0x76dfe21d, hkVersionRegistry::VERSION_COPY, "hkxTextureInplace", HK_NULL }, 
	{ 0x0217ef77, 0x548ff417, hkVersionRegistry::VERSION_COPY, "hkxTextureFile", HK_NULL },

	// behavior
	{ 0xba8f6319, 0x01902403, hkVersionRegistry::VERSION_COPY, "hkbEventDrivenModifier", HK_NULL },
	{ 0x1b0d2fac, 0xe5396080, hkVersionRegistry::VERSION_COPY, "hkbHandIkModifierHand", HK_NULL },
	{ 0xf5fb3115, 0xa054e9e1, hkVersionRegistry::VERSION_COPY, "hkbHandIkModifier", HK_NULL },
	{ 0x19618305, 0x5ba5955a, hkVersionRegistry::VERSION_COPY, "hkbHandIkControlsModifierHand", HK_NULL },
	{ 0x277ca1c1, 0xd87204b0, hkVersionRegistry::VERSION_COPY, "hkbHandIkControlsModifier", HK_NULL }, 
	{ 0x4a6c28da, 0xf3783030, hkVersionRegistry::VERSION_COPY, "hkbLookAtModifier", HK_NULL },
	{ 0xd941d03e, 0x50ed39ad, hkVersionRegistry::VERSION_COPY, "hkbPositionRelativeSelectorGenerator", HK_NULL },
	{ 0x9768b7c3, 0x267e83fb, hkVersionRegistry::VERSION_COPY, "hkbSplinePathGenerator", HK_NULL },
	{ 0xfec86404, 0x133ba407, hkVersionRegistry::VERSION_COPY, "hkbModifier", HK_NULL },
	{ 0x0160d338, 0x2de24779, hkVersionRegistry::VERSION_COPY, "hkbNode", HK_NULL },

	// physics
	{ 0x782e8ff3, 0xdae249ef, hkVersionRegistry::VERSION_COPY, "hkpListShape", UpdateListShape },
	{ 0x3fcd7295, 0xc89cb00f, hkVersionRegistry::VERSION_COPY, "hkpWorldCinfo", HK_NULL },

	{ 0x1d680046, 0x28343991, hkVersionRegistry::VERSION_COPY, "hkpBallSocketConstraintAtom", HK_NULL },

	{ 0x33f74135, 0x59c909e6, hkVersionRegistry::VERSION_COPY, "hkpBallAndSocketConstraintDataAtoms", HK_NULL },
	{ 0x8c4d3cf6, 0x54d28e04, hkVersionRegistry::VERSION_COPY, "hkpHingeConstraintDataAtoms", HK_NULL },
	{ 0xeb91c599, 0x51ada77a, hkVersionRegistry::VERSION_COPY, "hkpLimitedHingeConstraintDataAtoms", HK_NULL },
	{ 0xbb1e4ebc, 0x3f82bac4, hkVersionRegistry::VERSION_COPY, "hkpRagdollConstraintDataAtoms", HK_NULL },

	{ 0x77aabbef, 0x38424f73, hkVersionRegistry::VERSION_COPY, "hkpBallAndSocketConstraintData", HK_NULL },
	{ 0x502cad5a, 0x60c4d260, hkVersionRegistry::VERSION_COPY, "hkpHingeConstraintData", HK_NULL },
	{ 0xe863aa21, 0x208d3cd0, hkVersionRegistry::VERSION_COPY, "hkpLimitedHingeConstraintData", HK_NULL },
	{ 0x31f375e2, 0x45da0764, hkVersionRegistry::VERSION_COPY, "hkpRagdollConstraintData", HK_NULL },

	{ 0xdbf952ec, 0x011e7c11, hkVersionRegistry::VERSION_COPY, "hkAabbUint32", Update_Empty },
	{ 0xa8035513, 0xb5f0e6b1, hkVersionRegistry::VERSION_COPY, "hkpCollidableBoundingVolumeData", Update_Assert },
	{ 0x19e24f2b, 0xf204e3b7, hkVersionRegistry::VERSION_COPY, "hkpCollidable", Update_Assert },
	{ 0x4793379a, 0x959a580d, hkVersionRegistry::VERSION_COPY, "hkpEntity", Update_Empty },
	{ 0xcebb2443, 0x08c08009, hkVersionRegistry::VERSION_COPY, "hkpWorldObject", Update_Empty },

	// destruction  
	{ 0x73fb6a85, 0xb93e27b3, hkVersionRegistry::VERSION_COPY, "hkdDeformationController", HK_NULL },
	{ 0x44f0b86a, 0x17a9e82d, hkVersionRegistry::VERSION_COPY, "hkdWeaponBlueprint", HK_NULL },
	{ 0x48b4489c, 0xfce05ec9, hkVersionRegistry::VERSION_COPY, "hkdBreakableShapeConnection", Update_Assert }, // m_strength changed to m_contactArea
	{ 0x3f34a97d, 0xc62ede40, hkVersionRegistry::VERSION_COPY, "hkdBreakableShape", hkdBreakableShape_Update }, // m_findInitialContactPoints removed, m_flags added, Connection.m_strength changed to m_contactArea
	{ 0xe4f63498, 0x3185f6e6, hkVersionRegistry::VERSION_COPY, "hkdBreakableBody", HK_NULL }, // integrity system and integrity strength added, constraint strength
	{ 0x030a93a5, 0xf3b2e806, hkVersionRegistry::VERSION_COPY, "hkdBreakableBodyBlueprint", HK_NULL },
	{ 0x9fc3074f, 0x129e843a, hkVersionRegistry::VERSION_COPY, "hkdWoodFracture", HK_NULL },
	{ 0xf72db378, 0x50ae5c9e, hkVersionRegistry::VERSION_COPY, "hkdSliceFracture", HK_NULL },
	{ 0x67db3565, 0xa36460de, hkVersionRegistry::VERSION_COPY, "hkdSplitInHalfFracture", HK_NULL },
	{ 0xca5884da, 0xa55c2a5a, hkVersionRegistry::VERSION_COPY, "hkdFracture", HK_NULL },
	REMOVED("hkdConnectionStrengthTester"),

	// cloth
	{ 0x20ff8386, 0xdad0ecc7, hkVersionRegistry::VERSION_MANUAL, "hclMoveParticlesOperator", Update_hclMoveParticlesOperator}, // HCL-519, ordering of entries
	{ 0x33d133e8, 0x023a67cc, hkVersionRegistry::VERSION_MANUAL, "hclBlendSomeVerticesOperator", Update_hclBlendSomeVerticesOperator}, // HCL-503, HCL-695
	REMOVED("hclBlendSomeVerticesOperatorVertexTriple"), // HCL-695
	{ 0x463a9974, 0x99c1c86e, hkVersionRegistry::VERSION_COPY, "hclSimClothData", Update_hclSimClothData}, //HCL-704

	{ 0, 0, 0, HK_NULL, HK_NULL }
};	 

static const hkVersionRegistry::ClassRename s_renames[] =
{
	{ HK_NULL, HK_NULL }
};

#define HK_COMPAT_VERSION_FROM hkHavok600r1Classes
#define HK_COMPAT_VERSION_TO hkHavok610r1Classes
#define HK_COMPAT_OPTIONAL_UPDATE_FUNC update

extern hkVersionRegistry::UpdateDescription hkVersionUpdateDescription;

static hkResult HK_CALL update(
							   hkArray<hkVariant>& objectsInOut,
							   hkObjectUpdateTracker& tracker )
{
	const char* unwantedTypes[] = 
	{
		"hkAabbUint32",
		"hkpCollidableBoundingVolumeData",
		"hkpCollidable",
		"hkdBreakableShapeConnection",
		HK_NULL
	};
	removeUnwantedObjects( objectsInOut, unwantedTypes, tracker );
	return hkVersionUtil::updateSingleVersion( objectsInOut, tracker, hkVersionUpdateDescription, hkVersionRegistry::getInstance().getClassNameRegistry(HK_COMPAT_VERSION_TO::VersionString) );
}

#include<Common/Compat/Deprecated/Compat/hkCompat_Common.cxx>
#undef HK_COMPAT_VERSION_FROM
#undef HK_COMPAT_VERSION_TO

} // namespace hkCompat_hk600r1_hk610r1

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
