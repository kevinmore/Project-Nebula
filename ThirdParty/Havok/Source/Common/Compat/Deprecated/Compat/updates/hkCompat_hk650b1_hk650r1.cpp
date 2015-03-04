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

namespace hkCompat_hk650b1_hk650r1
{

	// this class helps us allocate objects from an hkClass and track them
	class Context
	{
		public:

			Context(hkArray<hkVariant>& objectsForVersioning, hkObjectUpdateTracker& tracker) : m_objectsForVersioning(objectsForVersioning), m_tracker(tracker)
			{
				HK_ASSERT(0x677856ec, s_context == HK_NULL);
				s_context = this;
			}

			~Context()
			{
				HK_ASSERT(0x6b472eec, s_context);
				updateObjectsForVersioning();
				s_context = HK_NULL;
			}

			static Context& getInstance()
			{
				HK_ASSERT(0x5d617873, s_context);
				return *s_context;
			}

			// allocate memory for new object and update tracker accordingly
			// usage: void* myObject = Context::getInstance().newObject(&MyClass);
			void* newObject(const hkClass* klass)
			{
				HK_ASSERT(0x1099cd9e, klass);
				hkVariant v;
				hkInt32 size = klass->getObjectSize();
				v.m_object = hkAllocateChunk<char>(size, HK_MEMORY_CLASS_SERIALIZE);
				hkString::memSet(v.m_object, 0, size);
				v.m_class = klass;
				m_createdObjects.pushBack(v);
				m_tracker.addChunk(v.m_object, size, HK_MEMORY_CLASS_SERIALIZE);
				m_tracker.addFinish(v.m_object, v.m_class->getName());
				return v.m_object;
			}

			// allocate raw memory
			void* allocateChunk( int size )
			{
				void* rawMemory = hkAllocateChunk<char>( size, HK_MEMORY_CLASS_SERIALIZE );
				m_tracker.addChunk( rawMemory, size, HK_MEMORY_CLASS_SERIALIZE );
				return rawMemory;
			}

			// Allocate a new object of the appropriate type for a member that is an object pointer.
			// Returns a variant for the created object
			hkVariant pointMemberToNewObject( const hkVariant& variant, const char* memberName, hkObjectUpdateTracker& tracker )
			{
				hkVariant childVariant;

				// create the object
				childVariant.m_class = variant.m_class->getMemberByName( memberName )->getClass();
				HK_ASSERT( 0xd781b735, childVariant.m_class != HK_NULL );
				childVariant.m_object = newObject( childVariant.m_class );
				
				// point the member to the new object
				hkClassMemberAccessor member( variant, memberName );
				member.asPointer() = childVariant.m_object;

				// update the tracker as the new object is referenced by another object now
				tracker.objectPointedBy( childVariant.m_object, member.getAddress() );

				return childVariant;
			}

			const hkArray<hkVariant>& getObjectsForVersioning() const { return m_objectsForVersioning; }

		private:

			void updateObjectsForVersioning() const
			{
				hkVariant* v = m_objectsForVersioning.expandBy(m_createdObjects.getSize());
				hkString::memCpy(v, m_createdObjects.begin(), hkSizeOf(hkVariant)*m_createdObjects.getSize());
			}

			hkArray<hkVariant> m_createdObjects;
			hkArray<hkVariant>& m_objectsForVersioning;
			hkObjectUpdateTracker& m_tracker;

			static Context* s_context;
	};

	Context* Context::s_context = HK_NULL;

#define REMOVED(TYPE) { 0,0, hkVersionRegistry::VERSION_REMOVED, TYPE, HK_NULL }
#define BINARY_IDENTICAL(OLDSIG,NEWSIG,TYPE) { OLDSIG, NEWSIG, hkVersionRegistry::VERSION_MANUAL, TYPE, HK_NULL }

	struct DummyArray
	{
		void* data;
		int size;
		int capacity;
	};

	static void Update_ignore( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
	}

	// allocate the memory for an array
	static void initArray( hkClassMemberAccessor& member, int numElements, hkObjectUpdateTracker& tracker )
	{
		DummyArray& dummyArray = *static_cast<DummyArray*>(member.getAddress());
		dummyArray.size = numElements;
		dummyArray.capacity = hkArray<char>::DONT_DEALLOCATE_FLAG | numElements;

		if( numElements > 0 )
		{
			int numBytes = numElements * member.getClassMember().getArrayMemberSize();
			dummyArray.data = hkAllocateChunk<char>( numBytes, HK_MEMORY_CLASS_SERIALIZE );
			tracker.addChunk( dummyArray.data, numBytes, HK_MEMORY_CLASS_SERIALIZE );
		}
	}

	static void Update_MassChangerModifierConstraintAtom( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkClassMemberAccessor oldFactorA(oldObj, "factorA");
		hkClassMemberAccessor newFactorA(newObj, "factorA");
		hkClassMemberAccessor oldFactorB(oldObj, "factorB");
		hkClassMemberAccessor newFactorB(newObj, "factorB");

		hkReal factorA = oldFactorA.asReal();
		hkReal factorB = oldFactorB.asReal();
		for (int i = 0; i < 4; i++)
		{
			newFactorA.asVector4().r[i] = factorA;
			newFactorB.asVector4().r[i] = factorB;
		}
	}

	/////////////
	// behavior
	/////////////

	static void versionTransitionInfoArray( hkClassMemberAccessor& oldArrayMember, hkClassMemberAccessor& newArrayMember, hkObjectUpdateTracker& tracker )
	{
		DummyArray& oldArray = *static_cast<DummyArray*>(oldArrayMember.getAddress());
		DummyArray& newArray = *static_cast<DummyArray*>(newArrayMember.getAddress());

		int sz = oldArrayMember.getClassMember().getArrayMemberSize();

		// the size should not have changed
		HK_ASSERT( 0xdf7a612b, sz == newArrayMember.getClassMember().getArrayMemberSize() );

		// copy the array data
		hkString::memCpy( newArray.data, oldArray.data, sz * oldArray.size );

		const hkClass* newElementClass = newArrayMember.getClassMember().getClass();
		int offsetOfTransitionEffect = newElementClass->getMemberByName( "transition" )->getOffset();
		int offsetOfCondition = newElementClass->getMemberByName( "condition" )->getOffset();

		for( int i = 0; i < oldArray.size; i++ )
		{
			char* newP = (char*)(newArray.data) + sz * i;

			// we need to track the pointers to the hkbTransitionEffect and hkbPredicate
			{
				char** addressOfTransitionEffect = reinterpret_cast<char**>( newP + offsetOfTransitionEffect );

				if ( *addressOfTransitionEffect != HK_NULL )
				{
					tracker.objectPointedBy( *addressOfTransitionEffect, addressOfTransitionEffect );
				}

				char** addressOfCondition = reinterpret_cast<char**>( newP + offsetOfCondition );

				if ( *addressOfCondition != HK_NULL )
				{
					tracker.objectPointedBy( *addressOfCondition, addressOfCondition );
				}
			}
		}
	}

	static void Update_hkbStateMachineStateInfo( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkClassMemberAccessor oldTransitions( oldObj, "transitions" );

		DummyArray& oldTransitionsArray = *static_cast<DummyArray*>(oldTransitions.getAddress());

		if ( oldTransitionsArray.size > 0 )
		{
			hkClassMemberAccessor newTransitions( newObj, "transitions" );

			// create a new TransitionInfoArray object and point m_transitions to it
			hkVariant transitionInfoArrayVariant = Context::getInstance().pointMemberToNewObject( newObj, "transitions", tracker );

			// get the actual array out of the TransitionInfoArray
			hkClassMemberAccessor newTransitionInfoArrayMember( transitionInfoArrayVariant, "transitions" );

			// allocate memory for the array
			initArray( newTransitionInfoArrayMember, oldTransitionsArray.size, tracker );

			// copy the array
			versionTransitionInfoArray( oldTransitions, newTransitionInfoArrayMember, tracker );
		}
	}

	static void Update_hkbVariableBindingSet( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// We need to go through all of the bindings and find any bindings to "enable" and 
		// store the index.

		hkClassMemberAccessor bindings( newObj, "bindings" );
		hkClassMemberAccessor indexOfBindingToEnable( newObj, "indexOfBindingToEnable" );

		DummyArray& bindingsArray = *static_cast<DummyArray*>(bindings.getAddress());
		
		hkVariant bindingVariant;
		bindingVariant.m_class = &bindings.getClassMember().getStructClass();
		int sizeOfBinding = bindingVariant.m_class->getObjectSize();

		for( int i = 0; i < bindingsArray.size; i++ )
		{
			bindingVariant.m_object = (char*)bindingsArray.data + ( i * sizeOfBinding );
			hkClassMemberAccessor memberPathAccessor( bindingVariant, "memberPath" );
			char* memberPath = memberPathAccessor.asCstring();

			if ( 0 == hkString::strCmp( memberPath, "enable" ) )
			{
				indexOfBindingToEnable.asInt32() = i;
				return;
			}
		}

		indexOfBindingToEnable.asInt32() = -1;
	}

	static void Update_hkpVehicleInstanceWheelInfo( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkClassMemberAccessor oldShapeKey( oldObj, "contactShapeKey" );
		hkClassMemberAccessor newShapeKeys( newObj, "contactShapeKey" );

		// static alloc'ed array
		newShapeKeys.asUint32(0) = oldShapeKey.asUint32();
		for (int i = 1; i < newShapeKeys.getClassMember().getCstyleArraySize(); ++i)
		{
			newShapeKeys.asUint32(i) = hkUint32(-1);
		}
	}

	static void Update_hkpVehicleInstance( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkClassMemberAccessor oldWheelsInfoArray(oldObj, "wheelsInfo");
		hkClassMemberAccessor newWheelsInfoArray(newObj, "wheelsInfo");

		hkClassMemberAccessor::SimpleArray& oldArray = oldWheelsInfoArray.asSimpleArray();
		hkClassMemberAccessor::SimpleArray& newArray = newWheelsInfoArray.asSimpleArray();
		HK_ASSERT( 0xad25585b, oldArray.size == newArray.size );
		hkVariant oldWheelInfoVariant = {HK_NULL, &oldWheelsInfoArray.object().getClass()};
		hkVariant newWheelInfoVariant = {HK_NULL, &newWheelsInfoArray.object().getClass()};
		hkInt32 oldWheelInfoSize = oldWheelInfoVariant.m_class->getObjectSize();
		hkInt32 newWheelInfoSize = newWheelInfoVariant.m_class->getObjectSize();
		for( int i = 0; i < oldArray.size; ++i )
		{
			oldWheelInfoVariant.m_object = static_cast<char*>(oldArray.data) + oldWheelInfoSize*i;
			newWheelInfoVariant.m_object = static_cast<char*>(newArray.data) + newWheelInfoSize*i;
			Update_hkpVehicleInstanceWheelInfo(oldWheelInfoVariant, newWheelInfoVariant, tracker);
		}
	}

	static void UpdateStridingTypeInTrianglesSubpart( hkVariant& obj )
	{
		hkClassMemberAccessor stridingType( obj, "stridingType" );

		// IndexStridingType has new enum value INDICES_INT8=1 inserted before INDICES_INT16
		// it shifts all the values > 0 by 1
		if( stridingType.asInt8() > 0 )
		{
			++stridingType.asInt8();
		}
	}

	static void Update_hkpExtendedMeshShapeTrianglesSubpart( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		UpdateStridingTypeInTrianglesSubpart(newObj);
	}

	static void Update_hkpExtendedMeshShape( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkClassMemberAccessor newEmbeddedTrianglesSubpart( newObj, "embeddedTrianglesSubpart" );
		hkVariant newSubpart = { newEmbeddedTrianglesSubpart.getAddress(), newEmbeddedTrianglesSubpart.getClassMember().getClass() };
		UpdateStridingTypeInTrianglesSubpart(newSubpart);

		hkClassMemberAccessor newTrianglesSubparts( newObj, "trianglesSubparts" );
		const hkClass* trianglesSubpartsClass = newTrianglesSubparts.getClassMember().getClass();
		hkClassMemberAccessor::SimpleArray& trianglesSubparts = newTrianglesSubparts.asSimpleArray();
		for( int i = 0; i < trianglesSubparts.size; ++i )
		{
			hkVariant subpartObj = { HK_NULL, trianglesSubpartsClass };
			subpartObj.m_object = static_cast<char*>(trianglesSubparts.data) + trianglesSubpartsClass->getObjectSize()*i;
			UpdateStridingTypeInTrianglesSubpart(subpartObj);
		}
	}

	// Cloth

	static void Update_hclSkinOperator ( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// HCL-839 : Now using 16 bit "start influence" indices
		hkClassMemberAccessor oldStartArray ( oldObj, "boneInfluenceStartPerVertex");
		hkClassMemberAccessor newStartArray ( newObj, "boneInfluenceStartPerVertex");

		const int size = newStartArray.asSimpleArray().size = oldStartArray.asSimpleArray().size;
		// allocate memory for the array
		initArray( newStartArray, size, tracker);

		// for loop
		for (int i=0; i<size; ++i)
		{
			reinterpret_cast<hkUint16*> (newStartArray.asSimpleArray().data)[i] = static_cast<hkUint16> (reinterpret_cast<hkUint32*> (oldStartArray.asSimpleArray().data) [i]);
		}

	}

	static void Update_hclMeshMeshDeformOperator  ( hkVariant& oldVar, hkVariant& newVar, hkObjectUpdateTracker& tracker )
	{
		// HCL-839 : Now using start and end vertex entries. For old assets, these should be set to 0 and numVertices-1
		// Number of vertices can be guessed from the number of entries in "triangleVertexStartForVertex" in those assets,
		// which contains numVertices+1 entries.
		hkClassAccessor oldObj (oldVar);
		hkClassAccessor newObj (newVar);

		newObj.member("startVertex").asUint32() = 0;
		newObj.member("endVertex").asUint32() = oldObj.member("triangleVertexStartForVertex").asSimpleArray().size - 2;

	}

	static void Update_hclToolNamedObjectReference( hkVariant& oldVar, hkVariant& newVar, hkObjectUpdateTracker& tracker )
	{
		hkClassAccessor oldObj (oldVar);
		hkClassAccessor newObj (newVar);

		newObj.member("hash").asUint32() = (hkUint32)oldObj.member("hash").asUlong();
	}

static const hkVersionRegistry::ClassAction s_updateActions[] =
{
	// variants
	{ 0x6728e4b7, 0x6728e4b7, hkVersionRegistry::VERSION_VARIANT, "hkaBoneAttachment", HK_NULL },
	{ 0xf598a34e, 0xf598a34e, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainer", HK_NULL },
	{ 0x853a899c, 0x853a899c, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainerNamedVariant", HK_NULL },
	{ 0xdce3ca6b, 0xdce3ca6b, hkVersionRegistry::VERSION_VARIANT, "hkMemoryResourceHandle", HK_NULL },
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
	{ 0x15d99dc6, 0x15d99dc6, hkVersionRegistry::VERSION_VARIANT, "hkbVariableValueSet", HK_NULL },
	{ 0xa71c409c, 0xa71c409c, hkVersionRegistry::VERSION_VARIANT, "hkdShape", HK_NULL },
	{ 0xf7925275, 0x1bbfdb97, hkVersionRegistry::VERSION_VARIANT|hkVersionRegistry::VERSION_COPY, "hkdBody", HK_NULL },
	{ 0x8e9b1727, 0x8e9b1727, hkVersionRegistry::VERSION_HOMOGENEOUSARRAY, "hkxVertexBuffer", HK_NULL },

	// Added second vcolor to padding:
	BINARY_IDENTICAL (0x34b6ba25, 0x6113c224, "hkxVertexP4N4C1T10"),
	BINARY_IDENTICAL (0x74c3397a, 0xeba9fa94, "hkxVertexP4N4C1T2"),
	BINARY_IDENTICAL (0xbd788073, 0x386012b0, "hkxVertexP4N4C1T6"),
	BINARY_IDENTICAL (0x84175c4a, 0x430501e7, "hkxVertexP4N4T4B4C1T10"),
	BINARY_IDENTICAL (0xce018bf0, 0xd57ff474, "hkxVertexP4N4T4B4C1T2"),
	BINARY_IDENTICAL (0xe0a86dbf, 0x72a6f526, "hkxVertexP4N4T4B4C1T6"),
	BINARY_IDENTICAL (0x8871a19c, 0x2dd90bc0, "hkxVertexP4N4T4B4W4I4C1T12"),
	BINARY_IDENTICAL (0x42f12f9d, 0x0e7c1052, "hkxVertexP4N4T4B4W4I4C1T4"),
	BINARY_IDENTICAL (0x14eef521, 0x5910bdbe, "hkxVertexP4N4T4B4W4I4C1T8"),
	BINARY_IDENTICAL (0x8c09d17a, 0x9239436e, "hkxVertexP4N4W4I4C1T12"),
	BINARY_IDENTICAL (0x5578b7b9, 0xe681948b, "hkxVertexP4N4W4I4C1T4"),
	BINARY_IDENTICAL (0xa50733d9, 0x10b369e7, "hkxVertexP4N4W4I4C1T8"),


	// Common
	{ 0x281423f6, 0x2a57a90a, hkVersionRegistry::VERSION_COPY, "hkUiAttribute", HK_NULL },

	// Physics
	{ 0xea8885d7, 0x568593cc, hkVersionRegistry::VERSION_COPY, "hkpAabbPhantom", HK_NULL },
	{ 0x8dfae56b, 0x9d1464fb, hkVersionRegistry::VERSION_COPY, "hkpCachingShapePhantom", HK_NULL },
	{ 0x8dfae56b, 0x9d1464fb, hkVersionRegistry::VERSION_COPY, "hkpSimpleShapePhantom", HK_NULL },
	{ 0xc92fe7b3, 0x5a8169ee, hkVersionRegistry::VERSION_COPY, "hkpEntity", HK_NULL },
	{ 0x6e70ee3f, 0x39409e2d, hkVersionRegistry::VERSION_COPY, "hkpWorldObject", HK_NULL },

	{ 0x28343991, 0x67a6e6ba, hkVersionRegistry::VERSION_COPY, "hkpBallSocketConstraintAtom", HK_NULL }, // new member added, members ordering changed
	{ 0x2802f3cb, 0xbedd5f22, hkVersionRegistry::VERSION_COPY, "hkpMassChangerModifierConstraintAtom", Update_MassChangerModifierConstraintAtom },
	BINARY_IDENTICAL( 0xed982d04, 0xd33c512e, "hkpConstraintAtom"), // New atom type added at the end of enumeration

	{ 0x59c909e6, 0x3e208629, hkVersionRegistry::VERSION_COPY, "hkpBallAndSocketConstraintDataAtoms", HK_NULL },
	{ 0x54d28e04, 0xd9d0ebf0, hkVersionRegistry::VERSION_COPY, "hkpHingeConstraintDataAtoms", HK_NULL },
	BINARY_IDENTICAL( 0x19a836b1, 0x47e00ae1, "hkpHingeLimitsDataAtoms"),
	{ 0x51ada77a, 0x207b4cda, hkVersionRegistry::VERSION_COPY, "hkpLimitedHingeConstraintDataAtoms", HK_NULL },
	BINARY_IDENTICAL( 0x71423c29, 0x55c24b79, "hkpBridgeAtoms"),
	BINARY_IDENTICAL( 0xb191e7d3, 0x03a0aa8e, "hkpPointToPlaneConstraintDataAtoms"),
	BINARY_IDENTICAL( 0xf7809c2d, 0xa19b09b6, "hkpPrismaticConstraintDataAtoms"),
	BINARY_IDENTICAL( 0xb0b4ce17, 0xecd4b4be, "hkpPulleyConstraintDataAtoms"),
	{ 0x3f82bac4, 0x407bf11b, hkVersionRegistry::VERSION_COPY, "hkpRagdollConstraintDataAtoms", HK_NULL },
	BINARY_IDENTICAL( 0xf4de43f4, 0x93990770, "hkpRagdollLimitsDataAtoms"),
	BINARY_IDENTICAL( 0xab648d8d, 0xd6b4fa0e, "hkpRotationalConstraintDataAtoms"),
	BINARY_IDENTICAL( 0x4616cbb4, 0x6177df03, "hkpStiffSpringConstraintDataAtoms"),
	BINARY_IDENTICAL( 0xdcf27bc0, 0xbe9b5f8b, "hkpWheelConstraintDataAtoms"),

	{ 0x38424f73, 0xfbcb20b8, hkVersionRegistry::VERSION_COPY, "hkpBallAndSocketConstraintData", HK_NULL },
	BINARY_IDENTICAL( 0xea22d4f9, 0x4b9c419, "hkpBallSocketChainData"),
	BINARY_IDENTICAL( 0x33eab5e, 0x9800d734, "hkpGenericConstraintData"),
	{ 0x60c4d260, 0xa8ef77c, hkVersionRegistry::VERSION_COPY, "hkpHingeConstraintData", HK_NULL },
	BINARY_IDENTICAL( 0xd023da1a, 0x406417ae, "hkpHingeLimitsData"),
	{ 0x208d3cd0, 0xd458bb24, hkVersionRegistry::VERSION_COPY, "hkpLimitedHingeConstraintData", HK_NULL },
	BINARY_IDENTICAL( 0x73937118, 0xa4bf9c94, "hkpMalleableConstraintData"),
	BINARY_IDENTICAL( 0xcf3f2f29, 0xca168aae, "hkpPointToPathConstraintData"),
	BINARY_IDENTICAL( 0xb3c7ee7f, 0x49757895, "hkpPointToPlaneConstraintData"),
	BINARY_IDENTICAL( 0x2c90d2b4, 0x4a138d43, "hkpPoweredChainData"),
	BINARY_IDENTICAL( 0xcf55cbd3, 0xbff614e0, "hkpPrismaticConstraintData"),
	BINARY_IDENTICAL( 0x8f4fcde2, 0x130b2d6, "hkpPulleyConstraintData"),
	{ 0x45da0764, 0x43c94a7f, hkVersionRegistry::VERSION_COPY, "hkpRagdollConstraintData", HK_NULL },
	BINARY_IDENTICAL( 0x93b97b48, 0x9f00917f, "hkpRagdollLimitsData"),
	BINARY_IDENTICAL( 0xec1fc462, 0x93d5253c, "hkpRotationalConstraintData"),
	BINARY_IDENTICAL( 0xfb925fd5, 0xf8bf3859, "hkpSerializedAgentNnEntry"),
	BINARY_IDENTICAL( 0x330795be, 0xef7de5b0, "hkpStiffSpringChainData"),
	BINARY_IDENTICAL( 0x808a3957, 0x27f20ebe, "hkpStiffSpringConstraintData"),
	BINARY_IDENTICAL( 0x94415572, 0x3dff463d, "hkpWheelConstraintData"),

	{ 0xacbe266e, 0x695c1674, hkVersionRegistry::VERSION_MANUAL, "hkpExtendedMeshShape", Update_hkpExtendedMeshShape },
	{ 0x36469972, 0x2bf65ec4, hkVersionRegistry::VERSION_MANUAL, "hkpExtendedMeshShapeTrianglesSubpart", Update_hkpExtendedMeshShapeTrianglesSubpart },
	BINARY_IDENTICAL( 0x5d634d20, 0x255df72d, "hkpExtendedMeshShapeSubpart"), // fixed enum references, COM-571

	{ 0x1e37d40d, 0x0e0d8c23, hkVersionRegistry::VERSION_COPY, "hkpStorageExtendedMeshShapeMeshSubpartStorage", HK_NULL },

	BINARY_IDENTICAL( 0x31affdcf, 0x4f4bcb62, "hkpMeshShape"), // made enum names unique
	BINARY_IDENTICAL( 0xd296d43b, 0x27336e5d, "hkpMeshShapeSubpart"), // changed enum name 

	REMOVED("hkpPairwiseCollisionFilter"),
	REMOVED("hkpPairwiseCollisionFilterCollisionPair"),

	{ 0x22c896d9, 0x8c1ad171, hkVersionRegistry::VERSION_COPY, "hkpVehicleInstance", Update_hkpVehicleInstance },
	{ 0x80ce3610, 0x99f693f0, hkVersionRegistry::VERSION_COPY, "hkpVehicleInstanceWheelInfo", Update_hkpVehicleInstanceWheelInfo },
	{ 0x4e6c0972, 0xba643b88, hkVersionRegistry::VERSION_COPY, "hkpWorldCinfo", HK_NULL },
	{ 0x98900a8a, 0x6f0be4b1, hkVersionRegistry::VERSION_COPY, "hkpBreakableConstraintData", HK_NULL },
	{ 0x95b74236, 0xd0e73ea6, hkVersionRegistry::VERSION_COPY, "hkpConstraintInstance", HK_NULL },

	// behavior
	BINARY_IDENTICAL( 0x6d076c86, 0x774632b, "hkbAttachmentSetup" ),
	{ 0x4b3e5b60, 0x84a96b43, hkVersionRegistry::VERSION_COPY, "hkbCharacterData", HK_NULL },
	{ 0x7d693b27, 0x3a2f4672, hkVersionRegistry::VERSION_COPY, "hkbCompiledExpressionSet", HK_NULL },
	{ 0x6645ff10, 0x5f860541, hkVersionRegistry::VERSION_COPY, "hkbCompiledExpressionSetToken", HK_NULL },
	{ 0xdb4959bb, 0xc03171b6, hkVersionRegistry::VERSION_COPY, "hkbDampingModifier", HK_NULL },
	{ 0x5a2a86ba, 0x29e3fdd2, hkVersionRegistry::VERSION_COPY, "hkbDemoConfig", HK_NULL },
	{ 0x8ba1e251, 0xfc825fae, hkVersionRegistry::VERSION_COPY, "hkbDemoConfigStickVariableInfo", HK_NULL },
	{ 0x06081496, 0x1ba8f825, hkVersionRegistry::VERSION_COPY, "hkbExpressionCondition", HK_NULL },
	{ 0x906f11a2, 0x2e86fbec, hkVersionRegistry::VERSION_COPY, "hkbGetHandleOnBoneModifier", HK_NULL },
	{ 0x5274b2ac, 0x5f99f6db, hkVersionRegistry::VERSION_COPY, "hkbProxyModifier", HK_NULL },
	{ 0x69da6ce1, 0x33fc628a, hkVersionRegistry::VERSION_COPY, "hkbRotateCharacterModifier", HK_NULL },
	{ 0x68966be4, 0x14a9d072, hkVersionRegistry::VERSION_COPY, "hkbStateMachineStateInfo", Update_hkbStateMachineStateInfo },
	{ 0x6a6beab7, 0xacb6cbc2, hkVersionRegistry::VERSION_COPY, "hkbVariableBindingSet", Update_hkbVariableBindingSet },
	BINARY_IDENTICAL( 0x1081c68a, 0xc0a5a37a, "hkbModifier" ),
	BINARY_IDENTICAL( 0xcc088ba1, 0xdc0159bf, "hkbNode" ),
	REMOVED( "hkbSillyEventPayload" ),

	// Destruction
	{ 0x3185f6e6, 0xf6d23796, hkVersionRegistry::VERSION_COPY, "hkdBreakableBody", HK_NULL },
	{ 0xbb39d162, 0xcfaf5796, hkVersionRegistry::VERSION_COPY, "hkdDeformationController", HK_NULL },
	{ 0xf7b2b646, 0x7a38ac49, hkVersionRegistry::VERSION_COPY, "hkdFlexibleJointController", HK_NULL },
	{ 0x897bd4c9, 0x72065c60, hkVersionRegistry::VERSION_COPY, "hkdFlexibleJointControllerJointDef", HK_NULL },
 	{ 0xb5eadd03, 0xcb15c1dd, hkVersionRegistry::VERSION_COPY, "hkdVoronoiFracture", HK_NULL },
	{ 0x8e16744d, 0xf696cd0b, hkVersionRegistry::VERSION_COPY, "hkdFracture", HK_NULL },

	// Cloth
	{ 0xd10ab1c8, 0x1b235b1e, hkVersionRegistry::VERSION_COPY, "hclSkinOperator", Update_hclSkinOperator }, // HCL-839
	{ 0x9d3ef807, 0x2a5e887b, hkVersionRegistry::VERSION_COPY, "hclMeshMeshDeformOperator", Update_hclMeshMeshDeformOperator }, // HCL-839
	{ 0xdc2aac4d, 0x3f0fe451, hkVersionRegistry::VERSION_COPY, "hclToolNamedObjectReference", Update_hclToolNamedObjectReference }, // HCL-863

	{ 0, 0, 0, HK_NULL, HK_NULL }
};	 

static const hkVersionRegistry::ClassRename s_renames[] =
{
	{ HK_NULL, HK_NULL }
};

static void Update_hclScratchBufferDefinition ( hkVariant& oldVar, hkVariant& newVar, hkObjectUpdateTracker& tracker )
{
	hkClassAccessor newObj (newVar);

	// Old scratch buffers couldn't contain triangles, so make sure they don't lie about it
	newObj.member("numTriangles").asUint32() = 0;
	
}

static const hkVersionRegistry::ClassAction s_scratchActions[] =
{
	// cloth
	{ 0x2b832fed, 0x2b832fed, hkVersionRegistry::VERSION_COPY, "hclBufferDefinition", Update_hclScratchBufferDefinition },

	{ 0, 0, 0, HK_NULL, HK_NULL }
};	 

static const hkVersionRegistry::ClassRename s_scratchRenames[] =
{
	{ "hclBufferDefinition", "hclScratchBufferDefinition" },
	{ HK_NULL, HK_NULL }
};

#define HK_COMPAT_VERSION_FROM hkHavok650b1Classes
#define HK_COMPAT_VERSION_TO hkHavok650r1Classes

hkVersionRegistry::UpdateDescription hkScratchVersionUpdateDescription(s_scratchRenames, s_scratchActions, & HK_COMPAT_VERSION_TO::hkHavokDefaultClassRegistry);

#define HK_COMPAT_OPTIONAL_UPDATE_FUNC update

extern hkVersionRegistry::UpdateDescription hkVersionUpdateDescription;


static hkResult HK_CALL update ( hkArray<hkVariant>& objectsInOut,
							    hkObjectUpdateTracker& tracker )
{
	Context context(objectsInOut, tracker);

	hkArray<hkVariant> scratchBuffers;
	hkArray<hkVariant> nonScratchObjects;
	nonScratchObjects.reserve(objectsInOut.getSize());

	for (int i=0; i<objectsInOut.getSize(); ++i)
	{
		if (hkString::strCmp (objectsInOut[i].m_class->getName(), "hclBufferDefinition")==0)
		{
			hkClassMemberAccessor typeMember(objectsInOut[i],"type");
			switch (typeMember.asUint32())
			{
				case 6: // Scratch_P:
				case 7: // Scratch_PN:
				case 8: // Scratch_PNTB:
					scratchBuffers.pushBack(objectsInOut[i]);
					break;
				default:
					nonScratchObjects.pushBackUnchecked(objectsInOut[i]);
					break;
			}
		}
		else
		{
			nonScratchObjects.pushBackUnchecked(objectsInOut[i]);
		}
	}

	objectsInOut.swap(nonScratchObjects);

	hkResult res = hkVersionUtil::updateSingleVersion( objectsInOut, tracker, hkVersionUpdateDescription, hkVersionRegistry::getInstance().getClassNameRegistry(HK_COMPAT_VERSION_TO::VersionString) );
	if( res == HK_SUCCESS )
	{
		res = hkVersionUtil::updateSingleVersion( scratchBuffers, tracker, hkScratchVersionUpdateDescription, hkVersionRegistry::getInstance().getClassNameRegistry(HK_COMPAT_VERSION_TO::VersionString) );
	}

	for (int i=0; i<scratchBuffers.getSize(); ++i)
	{
		objectsInOut.pushBackUnchecked(scratchBuffers[i]);
	}

	return res;
}

#include<Common/Compat/Deprecated/Compat/hkCompat_Common.cxx>
#undef HK_COMPAT_VERSION_FROM
#undef HK_COMPAT_VERSION_TO

} // namespace hkCompat_hk650b1_hk650r1

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
