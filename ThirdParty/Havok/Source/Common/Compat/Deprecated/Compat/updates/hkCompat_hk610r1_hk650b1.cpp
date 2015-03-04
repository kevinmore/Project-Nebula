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

namespace hkCompat_hk610r1_hk650b1
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

			void putVariableBindingSetObjectsLast()
			{
				// We take out all the variable bindings and put it at the end of the array.
				// We do this to make sure that when we are updating the variable bindings the nodes
				// to which these variable bindings belong are already versioned.
				hkArray<hkVariant> variableBindingSetObjects;
				for ( int i = 0; i < m_objectsForVersioning.getSize(); ++i )
				{
					hkVariant& v = m_objectsForVersioning[i];
					const hkClass* oldVariableBindingSetClass = getClass( v.m_class, "hkbVariableBindingSet" );

					if ( oldVariableBindingSetClass != HK_NULL )
					{
						variableBindingSetObjects.pushBack( v );
						m_objectsForVersioning.removeAt( i );
						--i;
					}
				}

				for ( int i = 0; i < m_objectsForVersioning.getSize(); ++i )
				{
					hkVariant& v = m_objectsForVersioning[i];
					const hkClass* oldNodeClass = getClass( v.m_class, "hkbNode" );

					if ( oldNodeClass != HK_NULL )
					{
						hkVariant var;
						var.m_class = oldNodeClass;
						var.m_object = v.m_object;

						hkClassMemberAccessor variableBindingSet( var, "variableBindingSet" );
						char* variableBinding = *(char**)variableBindingSet.getAddress();

						if ( variableBinding != HK_NULL )
						{
							m_variableBindingToNodeMap.insert( variableBinding, i );
						}
					}
				}
				
				for ( int i = 0; i < variableBindingSetObjects.getSize(); ++i )
				{
					m_objectsForVersioning.pushBack( variableBindingSetObjects[i] );
				}
			}

			const hkPointerMap<char*, int>& getVariableBindingToNodeMap() const { return m_variableBindingToNodeMap; }
			const hkArray<hkVariant>& getObjectsForVersioning() const { return m_objectsForVersioning; }
			hkPointerMap<void*, const hkClass*>& getNewObjectToClassMap() { return m_newObjectToClassMap; }
			hkPointerMap<void*, void*>& getOldObjectToNewObjectMap() { return m_oldObjectToNewObjectMap; }

		private:
			void updateObjectsForVersioning() const
			{
				hkVariant* v = m_objectsForVersioning.expandBy(m_createdObjects.getSize());
				hkString::memCpy(v, m_createdObjects.begin(), hkSizeOf(hkVariant)*m_createdObjects.getSize());
			}

			static const hkClass* getClass( const hkClass* klass, const char* className ) 
			{
				while ( klass != HK_NULL )
				{
					if ( 0 == hkString::strCmp( className, klass->getName() ) )
					{
						return klass;
					}

					klass = klass->getParent();
				}

				return HK_NULL;
			}

			hkArray<hkVariant> m_createdObjects;
			hkArray<hkVariant>& m_objectsForVersioning;
			hkObjectUpdateTracker& m_tracker;

			// This map stores the node index and the variable binding associated with the node at the node index.

			// The 6.1.0 version of behavior supported variable binding sets which not only stored variable binding
			// information of the root object but also of the child objects. Post 6.1.0 this feature is no longer available.
			// Now all the child objects store there own binding information. To fix versioning the following algorithm 
			// was used.

			// Before running class specific version functions we put all the variable binding set object at the end of the
			// m_objectsForVersion list. This results in the version of the nodes ( which contain variable binding set ) 
			// before the versioning of the variable binding set. Then when versioning each variable binding set we go
			// through each binding and if the binding belongs to the root node we copy the binding else we allocate variable
			// binding set in the child object and copy the child bindings. 

			hkPointerMap<char*, int> m_variableBindingToNodeMap;

			// A map between new objects pointers and new classes
			hkPointerMap<void*, const hkClass*> m_newObjectToClassMap;

			// A map between old object pointers and new object pointers
			hkPointerMap<void*, void*> m_oldObjectToNewObjectMap;

			static Context* s_context;
	};

	Context* Context::s_context = HK_NULL;

#define REMOVED(TYPE) { 0,0, hkVersionRegistry::VERSION_REMOVED, TYPE, HK_NULL }
#define BINARY_IDENTICAL(OLDSIG,NEWSIG,TYPE) { OLDSIG, NEWSIG, hkVersionRegistry::VERSION_MANUAL, TYPE, HK_NULL }

	static bool isBindable( const hkClass* klass )
	{
		while ( klass != HK_NULL )
		{
			if ( 0 == hkString::strCmp( "hkbBindable", klass->getName() ) )
			{
				return true;
			}

			klass = klass->getParent();
		}

		return false;
	}

	// Given a root variant and a member path root at it, traverse the data and find the hkbBindable that is
	// closest to the end of the path.  Returns the hkbBindable and a new path from it to the member.  The new path
	// is just a suffix of the input path, and points into the same memory.
	static void findPathFromBindable(	const hkVariant& rootVariant, 
										const char* memberPath, 
										void*& bindableOut, 
										const hkClass*& bindableClassOut, 
										const char*& pathFromBindableOut )
	{
		bindableOut = HK_NULL;
		pathFromBindableOut = HK_NULL;

		const hkClass* klass = rootVariant.m_class;
		void* rootObject = rootVariant.m_object;

		while ( memberPath != HK_NULL )
		{
			if ( isBindable( klass ) )
			{
				bindableOut = rootObject;
				pathFromBindableOut = memberPath;
				bindableClassOut = klass;
			}

			HK_ASSERT2( 0x6a712400, *memberPath != '\0', "the member path is empty" );
			HK_ASSERT2( 0x6a712401, *memberPath != '/', "there should not be a leading slash" );
			HK_ASSERT2( 0x6a712402, klass != HK_NULL, "didn't find class for object" );

			const char* memberName = memberPath;
			const char* colon = HK_NULL;

			// the length of the member string
			int len;

			// the array index, if any
			int index;

			// find the end of this member name, and a colon if there is one (for array indexing)
			{
				const char* p = memberPath;

				for( p = memberPath; ( (*p) != '\0' ) && ( (*p) != '/' ); p++ )
				{
					if ( *p == ':' )
					{
						HK_ASSERT2( 0x6a712403, colon == HK_NULL, "two colons found for same member" );
						colon = p;
					}
				}

				if ( colon != HK_NULL )
				{
					index = hkString::atoi( colon + 1 );
					len = int( colon - memberPath );
				}
				else
				{
					index = 0;
					len = int( p - memberPath );
				}

				HK_ASSERT2( 0x6a712404, len > 0, "the length of a member name should not be zero" );

				// figure out the memberPath for the next iteration, if any
				if ( *p == '\0' )
				{
					memberPath = HK_NULL;
				}
				else // *p == '/'
				{
					memberPath = p + 1;
				}
			}

			// We search for the member manually instead of using hkClass::getMemberByName()
			// because our substring is not necessarily null-terminated.

			int numMembers = klass->getNumMembers();

			for( int i = 0; i < numMembers; i++ )
			{
				const hkClassMember& member = klass->getMember( i );

				// these are carefully short-circuited to make sure that the member is not merely a prefix
				if ( !hkString::strNcmp( memberName, member.getName(), len ) && ( member.getName()[len] == '\0' ) )
				{
					HK_ASSERT2( 0x6a712405, member.getType() != hkClassMember::TYPE_ZERO, "can't traverse a +nosaved member" );

					void* memberData = reinterpret_cast<void*>( reinterpret_cast<char*>(rootObject) + member.getOffset() );

					if ( member.getType() == hkClassMember::TYPE_ARRAY )
					{
						// bounds checking on arrays
						{
							HK_ON_DEBUG( hkArray<char>* _a = reinterpret_cast<hkArray<char>*>( memberData ) );
							HK_ASSERT2( 0x6a712406, ( index >= 0 ) && ( index < _a->getSize() ), "array index out of bounds" );
						}

						switch( member.getSubType() )
						{
							case hkClassMember::TYPE_POINTER:
							{
								if ( memberPath == HK_NULL )
								{
									// return if the path is empty
									return;
								}
								else
								{
									// if there is more in the path, drill down into the object
									hkArray<void*>* array = reinterpret_cast<hkArray<void*>*>( memberData );
									void** rootObjectPtr = array->begin() + index;
									rootObject = *rootObjectPtr;
									klass = &(member.getStructClass());
								}
								break;
							}
							case hkClassMember::TYPE_STRUCT:
							{
								HK_ASSERT2( 0x6a712408, memberPath != HK_NULL, "path ended prematurely" );
								hkArray<char>* array = reinterpret_cast<hkArray<char>*>( memberData );
								rootObject = array->begin() + index * member.getArrayMemberSize();
								klass = &(member.getStructClass());
								break;
							}
							case hkClassMember::TYPE_VECTOR4:
							{
								return;
							}
							case hkClassMember::TYPE_QUATERNION:
							{
								return;
							}

							default:
							{
								return;
							}
						}
					}
					else
					{
						// handle C-style arrays
						int arraySize = member.getCstyleArraySize();

						if ( ( arraySize != 0 ) && ( index != 0 ) )
						{
							HK_ASSERT2( 0x6a712409, (index >= 0) && (index < arraySize), "array index out of bounds" );

							int memberSize = member.getSizeInBytes() / arraySize;
							memberData = reinterpret_cast<char*>(memberData) + ( index * memberSize );
						}

						switch( member.getType() )
						{
							case hkClassMember::TYPE_POINTER:
							{
								if ( memberPath == HK_NULL )
								{
									return;
								}
								else
								{
									//HK_ASSERT2( 0x6a71240a, false, "pointers in member paths are deprecated" );

									// if there is more in the path, drill down into the object
									void** nextObjectPtr = reinterpret_cast<void**>( memberData );
									rootObject = *nextObjectPtr;
									klass = &(member.getStructClass());
								}
								break;
							}
							case hkClassMember::TYPE_STRUCT:
							{
								HK_ASSERT2( 0x6a71240b, memberPath != HK_NULL, "path ended prematurely" );
								rootObject = memberData;
								klass = &(member.getStructClass());
								break;
							}
							case hkClassMember::TYPE_VECTOR4:
							{
								return;
							}
							case hkClassMember::TYPE_QUATERNION:
							{
								return;
							}
							default:
							{
								return;
							}
						}
					}

					// don't loop anymore since we found the member we were looking for
					break;
				}
			}
		}

		// make sure we found something
		HK_ASSERT2( 0x6a71240c, bindableOut != HK_NULL, "hkbBindable not found" );
	}

	struct DummyArray
	{
		void* data;
		int size;
		int capacity;
	};

	struct DummyMaterial
	{
		hkUint32 filterInfo;
		class hkHalf restitution;
		class hkHalf friction;
		hkUlong userData;
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
			hkString::memSet( dummyArray.data, 0, numBytes );
		}
	}

		// turns an hkArray<hkReal> into an hkbBoneWeightArray
	static void versionBoneWeightArray(	hkVariant& oldObj, const char* oldArrayMemberName, 
										hkVariant& newObj, const char* newBoneWeightArrayMemberName,
										hkObjectUpdateTracker& tracker )
	{				
		hkClassMemberAccessor oldBoneWeights(oldObj, oldArrayMemberName);

		DummyArray& oldBoneWeightArray = *static_cast<DummyArray*>(oldBoneWeights.getAddress());

		if ( oldBoneWeightArray.size > 0 )
		{
			hkClassMemberAccessor newBoneWeights(newObj, newBoneWeightArrayMemberName);

			// create a new hkbBoneWeightArray object and point m_boneWeights to it
			hkVariant boneWeightArrayVariant = Context::getInstance().pointMemberToNewObject( newObj, newBoneWeightArrayMemberName, tracker );

			// get the actual array out of the hkbBoneWeightArray
			hkClassMemberAccessor newBoneWeightArrayMember( boneWeightArrayVariant, "boneWeights" );

			// allocate memory for the array
			initArray( newBoneWeightArrayMember, oldBoneWeightArray.size, tracker );

			DummyArray& newBoneWeightArray = *static_cast<DummyArray*>(newBoneWeightArrayMember.getAddress());

			// copy the bone weights
			hkString::memCpy( newBoneWeightArray.data, oldBoneWeightArray.data, sizeof(hkReal) * oldBoneWeightArray.size );
		}
	}

	// turns an hkArray<hkInt16> into an hkbBoneIndexArray
	static void versionBoneIndexArray(	hkVariant& oldObj, const char* oldArrayMemberName, 
										hkVariant& newObj, const char* newBoneIndicesArrayMemberName,
										hkObjectUpdateTracker& tracker )
	{				
		hkClassMemberAccessor oldBoneIndices(oldObj, oldArrayMemberName);

		DummyArray& oldBoneIndicesArray = *static_cast<DummyArray*>(oldBoneIndices.getAddress());

		if ( oldBoneIndicesArray.size > 0 )
		{
			hkClassMemberAccessor newBoneIndices(newObj, newBoneIndicesArrayMemberName);

			// create a new hkbBoneWeightArray object and point m_boneWeights to it
			hkVariant boneWeightArrayVariant = Context::getInstance().pointMemberToNewObject( newObj, newBoneIndicesArrayMemberName, tracker );

			// get the actual array out of the hkbBoneWeightArray
			hkClassMemberAccessor newBoneIndicesArrayMember( boneWeightArrayVariant, "boneIndices" );

			// allocate memory for the array
			initArray( newBoneIndicesArrayMember, oldBoneIndicesArray.size, tracker );

			DummyArray& newBoneIndicesArray = *static_cast<DummyArray*>(newBoneIndicesArrayMember.getAddress());

			// copy the bone weights
			hkString::memCpy( newBoneIndicesArray.data, oldBoneIndicesArray.data, sizeof(hkReal) * oldBoneIndicesArray.size );
		}
	}

	static void Update_hkbTimerModifier( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkClassMemberAccessor oldEventId(oldObj, "eventIdToSend");
		hkClassMemberAccessor newEvent(newObj, "alarmEvent");

		newEvent.member( "id" ).asInt32() = oldEventId.asInt32();
	}

	static void Update_hkbFootIkModifier( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkClassMemberAccessor oldLegs(oldObj, "legs");
		hkClassMemberAccessor newLegs(newObj, "legs");

		DummyArray& oldLegsArray = *static_cast<DummyArray*>(oldLegs.getAddress());
		DummyArray& newLegsArray = *static_cast<DummyArray*>(newLegs.getAddress());

		int numLegs = oldLegsArray.size;

		int oldLegSize = oldLegs.getClassMember().getStructClass().getObjectSize();
		int newLegSize = newLegs.getClassMember().getStructClass().getObjectSize();

		char* oldLegData = static_cast<char*>( oldLegsArray.data );
		char* newLegData = static_cast<char*>( newLegsArray.data );

		// loop through all of the legs and copy the old event ID to the ID of the new event
		for( int i = 0; i < numLegs; i++ )
		{
			hkClassAccessor oldLeg( oldLegData + i * oldLegSize, &oldLegs.getClassMember().getStructClass() );
			hkClassAccessor newLeg( newLegData + i * newLegSize, &newLegs.getClassMember().getStructClass() );

			newLeg.member("ungroundedEvent").member("id").asInt32() = oldLeg.member( "ungroundedEventId" ).asInt32();
		}
	}

	static void Update_hkbAttachmentModifier( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkClassAccessor oldAM( oldObj );
		hkClassAccessor newAM( newObj );

		// these members were changed from an event ID to an hkbEventProperty
		newAM.member( "sendToAttacherOnAttach" ).member( "id" ).asInt32() = oldAM.member( "sendToAttacherOnAttach" ).asInt32();
		newAM.member( "sendToAttacheeOnAttach" ).member( "id" ).asInt32() = oldAM.member( "sendToAttacheeOnAttach" ).asInt32();
		newAM.member( "sendToAttacherOnDetach" ).member( "id" ).asInt32() = oldAM.member( "sendToAttacherOnDetach" ).asInt32();
		newAM.member( "sendToAttacheeOnDetach" ).member( "id" ).asInt32() = oldAM.member( "sendToAttacheeOnDetach" ).asInt32();
	}

	static void Update_hkbSplinePathGenerator( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// an event ID became an hkbEventProperty
		hkClassMemberAccessor oldEventId(oldObj, "pathEndEventId");
		hkClassMemberAccessor newEvent(newObj, "pathEndEvent");

		newEvent.member( "id" ).asInt32() = oldEventId.asInt32();
	}

	static void Update_hkbPositionRelativeSelectorGenerator( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// an event ID became an hkbEventProperty
		hkClassMemberAccessor oldEventId(oldObj, "fixPositionEventId");
		hkClassMemberAccessor newEvent(newObj, "fixPositionEvent");

		newEvent.member( "id" ).asInt32() = oldEventId.asInt32();
	}

	static void Update_hkbCatchFallModifier( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// an event ID became an hkbEventProperty
		hkClassMemberAccessor oldEventId(oldObj, "catchFallDoneEventId");
		hkClassMemberAccessor newEvent(newObj, "catchFallDoneEvent");

		newEvent.member( "id" ).asInt32() = oldEventId.asInt32();

		versionBoneIndexArray( oldObj, "spineIndices", newObj, "spineIndices", tracker );
	}

	static void Update_hkbCheckRagdollSpeedModifier( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// an event ID became an hkbEventProperty
		hkClassMemberAccessor oldEventId(oldObj, "eventToSend");
		hkClassMemberAccessor newEvent(newObj, "eventToSend");

		newEvent.member( "id" ).asInt32() = oldEventId.asInt32();
	}

	static void Update_hkbMoveBoneTowardTargetModifier( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// an event ID became an hkbEventProperty
		hkClassMemberAccessor oldEventId(oldObj, "eventToSendWhenTargetReached");
		hkClassMemberAccessor newEvent(newObj, "eventToSendWhenTargetReached");

		newEvent.member( "id" ).asInt32() = oldEventId.asInt32();
	}

	static void Update_hkbTargetRigidBodyModifier( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkClassAccessor oldMod( oldObj );
		hkClassAccessor newMod( newObj );

		newMod.member( "eventToSend" ).member( "id" ).asInt32() = oldMod.member( "eventToSend" ).asInt32();
		newMod.member( "eventToSendToTarget" ).member( "id" ).asInt32() = oldMod.member( "eventToSendToTarget" ).asInt32();
		newMod.member( "closeToTargetEvent" ).member( "id" ).asInt32() = oldMod.member( "closeToTargetEventId" ).asInt32();
	}

	static void Update_hkbDetectCloseToGroundModifier( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// an event ID became an hkbEventProperty
		hkClassMemberAccessor oldEventId(oldObj, "closeToGroundEventId");
		hkClassMemberAccessor newEvent(newObj, "closeToGroundEvent");

		newEvent.member( "id" ).asInt32() = oldEventId.asInt32();
	}

	static void Update_hkbBehaviorGraphData( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		HK_ASSERT(0xa892b634, newObj.m_class->getMemberByName("variableInitialValues"));
		hkClassMemberAccessor newInitialValues(newObj, "variableInitialValues");

		// create a new hkbVariableValueSet object and point m_variableInitialValues to it
		hkVariant initialValuesVariant = Context::getInstance().pointMemberToNewObject( newObj, "variableInitialValues", tracker );

		// now we have initial values of the appropriate type, we can fill it in with the data
		{
			hkClassMemberAccessor newWordValues( initialValuesVariant, "wordVariableValues" );
			hkClassMemberAccessor newQuadValues( initialValuesVariant, "quadVariableValues" );
			hkClassMemberAccessor oldQuadValues( oldObj, "quadVariableInitialValues" );
			hkClassMemberAccessor oldVariableInfos( oldObj, "variableInfos" );

			DummyArray& newWordValuesArray = *static_cast<DummyArray*>(newWordValues.getAddress());
			DummyArray& oldVariableInfosArray = *static_cast<DummyArray*>(oldVariableInfos.getAddress());

			int numWords = oldVariableInfosArray.size;
			
			if ( numWords > 0 )
			{
				initArray( newWordValues, numWords, tracker );

				struct OldVariableInfoStruct
				{
					int initialValue;
					hkInt8 type;
				};
				
				OldVariableInfoStruct* oldVariableInfoData = (OldVariableInfoStruct*)(oldVariableInfosArray.data);
				int* newWordsData = (int*)(newWordValuesArray.data);
				
				for( int i = 0; i < numWords; i++ )
				{
					newWordsData[i] = oldVariableInfoData[i].initialValue;
				}
			}

			DummyArray& oldQuadValuesArray = *static_cast<DummyArray*>(oldQuadValues.getAddress());
			int numQuads = oldQuadValuesArray.size;

			HK_ASSERT( 0x56173b7a, numWords >= numQuads );

			if ( numQuads > 0 )
			{
				initArray( newQuadValues, numQuads, tracker );

				DummyArray& newQuadValuesArray = *static_cast<DummyArray*>(newQuadValues.getAddress());
				hkVector4* newQuadData = (hkVector4*)( newQuadValuesArray.data );
				hkVector4* oldQuadData = (hkVector4*)( oldQuadValuesArray.data );

				for( int i = 0; i < numQuads; i++ )
				{
					newQuadData[i] = oldQuadData[i];
				}
			}

			// initialize the variants array
			{
				hkClassMemberAccessor newVariantValues( initialValuesVariant, "variantVariableValues" );

				initArray( newVariantValues, numQuads, tracker );
			}
		}
	}

	static void Update_hkbBlenderGeneratorChild( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		Context::getInstance().getOldObjectToNewObjectMap().insert( oldObj.m_object, newObj.m_object );
		Context::getInstance().getNewObjectToClassMap().insert( newObj.m_object, newObj.m_class );
		versionBoneWeightArray( oldObj, "boneWeights", newObj, "boneWeights", tracker );
	}

	static void Update_hkbStateMachineTransitionInfo( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		HK_ASSERT2( 0x567ad561, false, "this should not be called as it is done elsewhere" );
	}

	static void versionTransitionInfoArray( hkClassMemberAccessor& oldArrayMember, hkClassMemberAccessor& newArrayMember, hkObjectUpdateTracker& tracker )
	{
		DummyArray& oldArray = *static_cast<DummyArray*>(oldArrayMember.getAddress());
		DummyArray& newArray = *static_cast<DummyArray*>(newArrayMember.getAddress());

		int oldSz = oldArrayMember.getClassMember().getArrayMemberSize();
		int newSz = newArrayMember.getClassMember().getArrayMemberSize();

		const hkClass* newElementClass = newArrayMember.getClassMember().getClass();
		int offsetOfTransitionEffect = newElementClass->getMemberByName( "transition" )->getOffset();
		int offsetOfCondition = newElementClass->getMemberByName( "condition" )->getOffset();

		for( int i = 0; i < oldArray.size; i++ )
		{
			char* oldP = (char*)(oldArray.data) + oldSz * i;
			char* newP = (char*)(newArray.data) + newSz * i;

			// the new TransitionInfo is a prefix of the old one, so we copy the new size of bytes over.
			hkString::memCpy( newP, oldP, newSz );

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
		hkClassMemberAccessor newTransitions( newObj, "transitions" );

		versionTransitionInfoArray( oldTransitions, newTransitions, tracker );
	}

	static void Update_hkbStateMachine( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkClassMemberAccessor oldGlobalTransitions( oldObj, "globalTransitions" );

		DummyArray& oldGlobalTransitionsArray = *static_cast<DummyArray*>(oldGlobalTransitions.getAddress());

		if ( oldGlobalTransitionsArray.size > 0 )
		{
			hkClassMemberAccessor newWildcardTransitions( newObj, "wildcardTransitions" );

			// create a new TransitionInfoArray object and point m_wildcardTransitions to it
			hkVariant transitionInfoArrayVariant = Context::getInstance().pointMemberToNewObject( newObj, "wildcardTransitions", tracker );

			// get the actual array out of the TransitionInfoArray
			hkClassMemberAccessor newTransitionInfoArrayMember( transitionInfoArrayVariant, "transitions" );

			// allocate memory for the array
			initArray( newTransitionInfoArrayMember, oldGlobalTransitionsArray.size, tracker );

			// copy the array
			versionTransitionInfoArray( oldGlobalTransitions, newTransitionInfoArrayMember, tracker );
		}
	}

	static void Update_hkbClipGenerator( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkClassMemberAccessor oldTriggers( oldObj, "triggers" );

		DummyArray& oldTriggersArray = *static_cast<DummyArray*>(oldTriggers.getAddress());

		if ( oldTriggersArray.size > 0 )
		{
			hkVariant newTriggersVariant = Context::getInstance().pointMemberToNewObject( newObj, "triggers", tracker );

			// get the actual array for the new triggers
			hkClassMemberAccessor newTriggers( newTriggersVariant, "triggers" );

			// allocate memory for the array
			initArray( newTriggers, oldTriggersArray.size, tracker );

			DummyArray& newTriggersArray = *static_cast<DummyArray*>(newTriggers.getAddress());

			// the triggers themselves are binary identical
			HK_ASSERT( 0x458ac651, oldTriggers.getClassMember().getArrayMemberSize() == newTriggers.getClassMember().getArrayMemberSize() );
			const int triggerSize = newTriggers.getClassMember().getArrayMemberSize();
			hkString::memCpy( newTriggersArray.data, oldTriggersArray.data, oldTriggersArray.size * triggerSize );
		}
	}

	static void Update_hkbPoweredRagdollControlsModifier( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		versionBoneWeightArray( oldObj, "boneWeights", newObj, "boneWeights", tracker );
	}

	static void Update_hkbKeyframeBonesModifier ( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		versionBoneIndexArray( oldObj, "keyframedBonesList", newObj, "keyframedBonesList", tracker );
	}

	static void Update_hkbRigidBodyRagdollModifier( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		versionBoneIndexArray( oldObj, "keyframedBonesList", newObj, "keyframedBonesList", tracker );
	}

	static void Update_hkbJigglerGroup( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		Context::getInstance().getOldObjectToNewObjectMap().insert( oldObj.m_object, newObj.m_object );
		Context::getInstance().getNewObjectToClassMap().insert( newObj.m_object, newObj.m_class );
		versionBoneIndexArray( oldObj, "boneIndices", newObj, "boneIndices", tracker );
	}

	
	static void Update_hkbVariableBindingSet( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		const hkArray<hkVariant>& objectsForVersioning = Context::getInstance().getObjectsForVersioning();
		
		hkClassMemberAccessor oldBindings( oldObj, "bindings" );
		DummyArray& oldBindingsArray = *static_cast<DummyArray*>(oldBindings.getAddress());

		if ( oldBindingsArray.size > 0 )
		{
			hkClassMemberAccessor newBindings( newObj, "bindings" );
			DummyArray& newBindingsArray = *static_cast<DummyArray*>(newBindings.getAddress());

			const hkPointerMap<char*, int>& variableBindingToNodeMap = Context::getInstance().getVariableBindingToNodeMap();
			
			int nodeIndex = -1;
			nodeIndex = variableBindingToNodeMap.getWithDefault( (char*)oldObj.m_object, nodeIndex );

			HK_ASSERT2(0x23824ade, nodeIndex != -1, "For every binding set there has to be a node" );

			const hkVariant& newNodeVariant = objectsForVersioning[nodeIndex];

			int oldSz = oldBindings.getClassMember().getArrayMemberSize();
			int newSz = newBindings.getClassMember().getArrayMemberSize();

			int objectBindingSize = 0;

			hkPointerMap<void*, const hkClass*>& newObjectToClassMap = Context::getInstance().getNewObjectToClassMap();
			hkPointerMap<void*, void*>& oldObjectToNewObjectMap = Context::getInstance().getOldObjectToNewObjectMap();

			hkVariant oldBindingVariant;
			oldBindingVariant.m_class = oldBindings.getClassMember().getClass();

			for ( int i = 0; i < oldBindingsArray.size; ++i )
			{
				oldBindingVariant.m_object = (char*)(oldBindingsArray.data) + oldSz * i;

				hkVariant localNodeVariant;
				{		
					hkClassMemberAccessor oldObjectAccessor( oldBindingVariant, "object" );
					char* oldObject = *(char**)oldObjectAccessor.getAddress();

					// Any pointer that we get from the old variant are all old and we need it to map it to new 
					// pointers. While mapping old pointers to new pointers if we find that old pointers have no 
					// corresponding new pointers then it means that the old pointers point to the root object and
					// we get this directly from the newNodeVariant.

					void* newObject = HK_NULL;
					newObject = oldObjectToNewObjectMap.getWithDefault( oldObject, newObject );

					if ( newObject == HK_NULL )
					{
						localNodeVariant.m_class = newNodeVariant.m_class;
						localNodeVariant.m_object = newNodeVariant.m_object;
					}
					else
					{
						const hkClass* newClass = HK_NULL;
						newClass = newObjectToClassMap.getWithDefault( (char*)newObject, newClass );

						localNodeVariant.m_class = newClass;
						localNodeVariant.m_object = newObject;
					}
				}

				hkClassMemberAccessor oldMemberPathAccessor( oldBindingVariant, "memberPath" );

				int oldMemberPathLength = hkString::strLen( *(char**)oldMemberPathAccessor.getAddress() );
				hkStringBuf oldMemberPath( *(char**)oldMemberPathAccessor.getAddress(), oldMemberPathLength );

				if (	( 0 == hkString::strCmp( newNodeVariant.m_class->getName(), "hkbBlenderGenerator" ) ) ||
						( 0 == hkString::strCmp( newNodeVariant.m_class->getName(), "hkbPoweredRagdollControlsModifier" ) ) )
				{
					oldMemberPath.replace( "boneWeights", "boneWeights/boneWeights", hkStringBuf::REPLACE_ONE );
				}
				else if (	( 0 == hkString::strCmp( newNodeVariant.m_class->getName(), "hkbRigidBodyRagdollModifier" ) ) ||
							( 0	== hkString::strCmp( newNodeVariant.m_class->getName(), "hkbKeyframeBonesModifier" ) ) )
				{
					oldMemberPath.replace( "keyframedBonesList", "keyframedBonesList/boneIndices", hkStringBuf::REPLACE_ONE );
				}
				else if ( 0 == hkString::strCmp( newNodeVariant.m_class->getName(), "hkbJigglerModifier" ) )
				{
					oldMemberPath.replace( "boneIndices", "boneIndices/boneIndices", hkStringBuf::REPLACE_ONE );
				}

				void* objectContainingMember = HK_NULL;
				const hkClass* classOfObjectContainingMember = HK_NULL;
				const char* newMemberPath;

				findPathFromBindable(	localNodeVariant, 
										oldMemberPath.cString(), 
										objectContainingMember,
										classOfObjectContainingMember,
										newMemberPath );				

				// If the node variant object and the object containing the member are the same then we just need to
				// copy the binding.
				if ( newNodeVariant.m_object == objectContainingMember )
				{
					hkString::memCpy(	(char*)(newBindingsArray.data) + (newSz * objectBindingSize), 
										(char*)(newBindingsArray.data) + (newSz * i),
										newSz  );

					objectBindingSize++;
				}
				else
				{
					// If the input and the output objects are not the same then the binding belongs to the child object.
					// The child object pointer is the old pointer and we need it to map it to the new pointer.
					if ( localNodeVariant.m_object != objectContainingMember )
					{
						objectContainingMember = oldObjectToNewObjectMap.getWithDefault( objectContainingMember, objectContainingMember );
					}

					localNodeVariant.m_object = objectContainingMember;
					localNodeVariant.m_class = classOfObjectContainingMember;

					hkClassMemberAccessor childBindingSet( localNodeVariant, "variableBindingSet" );

					if ( HK_NULL == childBindingSet.asPointer() )
					{
						// create a new variable binding set object and point m_variableBindingSet to it
						hkVariant childBindingSetVariant = Context::getInstance().pointMemberToNewObject( localNodeVariant, "variableBindingSet", tracker );

						// get the bindings array out of the m_variableBindingSet;
						hkClassMemberAccessor childBindings( childBindingSetVariant, "bindings" );

						// allocate memory for the array
						initArray( childBindings, oldBindingsArray.size, tracker );

						DummyArray& childBindingsArray = *static_cast<DummyArray*>(childBindings.getAddress());
						childBindingsArray.size = 0;
					}

					hkVariant childBindingSetVariant;
					childBindingSetVariant.m_class = childBindingSet.getClassMember().getClass();
					childBindingSetVariant.m_object = childBindingSet.asPointer();

					hkClassMemberAccessor childBindings( childBindingSetVariant, "bindings" );

					DummyArray& childBindingsArray = *static_cast<DummyArray*>(childBindings.getAddress());

					hkString::memCpy(	(char*)(childBindingsArray.data) + newSz * childBindingsArray.size, 
										(char*)(newBindingsArray.data) + (newSz * i), 
										newSz );

					hkVariant newChildBindingVariant;
					newChildBindingVariant.m_object = (char*)(childBindingsArray.data) + newSz * childBindingsArray.size;
					newChildBindingVariant.m_class = childBindings.getClassMember().getClass();
					
					hkClassMemberAccessor memberPathAccesessor( newChildBindingVariant, "memberPath" );

					const int memberPathLength = hkString::strLen( newMemberPath );

					char* correctMemberPath = *(char**)memberPathAccesessor.getAddress();
					correctMemberPath = (char*)Context::getInstance().allocateChunk( memberPathLength + 1 );
					hkString::memCpy( correctMemberPath, newMemberPath, memberPathLength );
					correctMemberPath[memberPathLength] = '\0';
					childBindingsArray.size++;
				}
			}

			newBindingsArray.size = objectBindingSize;
		}
	}

	static void Update_Assert( hkVariant& oldVar, hkVariant& newVar, hkObjectUpdateTracker& tracker )
	{
		HK_ASSERT2(0xad81021a, false, "This object cannot is not expected to be directly updated by versioning.");
	}

	static void Update_TypedBroadPhaseHandle( hkVariant& oldVar, hkVariant& newVar, hkObjectUpdateTracker& tracker )
	{
		// hkpCollidableQualityType was updated. 
		// All values shifted down by one (HK_COLLIDABLE_QUALITY_INVALID was 0 and is -1)
		// New value added (HK_COLLIDABLE_QUALITY_DEBRIS_SIMPLE_TOI between HK_COLLIDABLE_QUALITY_DEBRIS and HK_COLLIDABLE_QUALITY_MOVING)

		hkClassMemberAccessor oldQualityType(oldVar, "objectQualityType");
		hkClassMemberAccessor newQualityType(newVar, "objectQualityType");

		hkUint16& oldType = oldQualityType.asUint16();
		hkInt8& newType = newQualityType.asInt8();

		HK_ASSERT2(0xad810221, oldType < 100, "Old collidable quality type invalid.");

		if (oldType <= 3)
		{
			newType = hkInt8(oldType) - 1;
		}
		else
		{
			newType = hkInt8(oldType);
		}
	}

	static void Update_Collidable( hkVariant& oldVar, hkVariant& newVar, hkObjectUpdateTracker& tracker )
	{
		// hkpCollidableQualityType was updated.
		hkClassMemberAccessor oldTypedBroadPhaseHandleMember(oldVar, "broadPhaseHandle");
		hkClassMemberAccessor newTypedBroadPhaseHandleMember(newVar, "broadPhaseHandle");
		hkVariant oldObj = {oldTypedBroadPhaseHandleMember.getAddress(), oldTypedBroadPhaseHandleMember.getClassMember().getClass()};
		hkVariant newObj = {newTypedBroadPhaseHandleMember.getAddress(), newTypedBroadPhaseHandleMember.getClassMember().getClass()};
		Update_TypedBroadPhaseHandle(oldObj, newObj, tracker);
	}

	static void Update_WorldObject( hkVariant& oldVar, hkVariant& newVar, hkObjectUpdateTracker& tracker )
	{
		// hkpCollidableQualityType was updated.
		hkClassMemberAccessor oldCollidableMember(oldVar, "collidable");
		hkClassMemberAccessor newCollidableMember(newVar, "collidable");
		hkVariant oldObj = {oldCollidableMember.getAddress(), oldCollidableMember.getClassMember().getClass()};
		hkVariant newObj = {newCollidableMember.getAddress(), newCollidableMember.getClassMember().getClass()};
		Update_Collidable(oldObj, newObj, tracker);
	}

	static void Update_ConstraintInstance( hkVariant& oldVar, hkVariant& newVar, hkObjectUpdateTracker& tracker )
	{
		hkClassAccessor oldConstraintInstance(oldVar);
		hkClassAccessor newConstraintInstance(newVar);

		hkClassMemberAccessor oldPriorityAccessor = oldConstraintInstance.member("priority");
		hkClassMemberAccessor newPriorityAccessor = newConstraintInstance.member("priority");

		hkUint8& oldPriority = oldPriorityAccessor.asUint8();
		hkUint8& newPriority = newPriorityAccessor.asUint8();

		const hkClassEnum* oldPriorityEnum = oldConstraintInstance.getClass().getEnumByName("ConstraintPriority");
		const hkClassEnum* newPriorityEnum = oldConstraintInstance.getClass().getEnumByName("ConstraintPriority");

		const char* name;
		oldPriorityEnum->getNameOfValue(oldPriority, &name);
		int newValue;
		newPriorityEnum->getValueOfName(name, &newValue);
		newPriority = hkUint8(newValue);
	}

	// CLOTH UPDATES

	static void Update_hclTransitionConstraintSetPerParticle ( hkVariant& oldVar, hkVariant& newVar, hkObjectUpdateTracker& tracker )
	{
		hkClassAccessor oldObj (oldVar);
		hkClassAccessor newObj (newVar);

		newObj.member("toAnimDelay").asReal() = oldObj.member("particleDelay").asReal();
		newObj.member("toSimDelay").asReal() = oldObj.member("particleDelay").asReal();
		newObj.member("toSimMaxDistance").asReal() = 1.0f;
	}

	static void Update_hclTransitionConstraintSet ( hkVariant& oldVar, hkVariant& newVar, hkObjectUpdateTracker& tracker )
	{
		hkClassAccessor oldObj (oldVar);
		hkClassAccessor newObj (newVar);

		newObj.member("toAnimPeriod").asReal() = oldObj.member("transitionPeriod").asReal();
		newObj.member("toSimPeriod").asReal() = oldObj.member("transitionPeriod").asReal();
		newObj.member("toAnimPlusDelayPeriod").asReal() = oldObj.member("transitionPlusDelayPeriod").asReal();
		newObj.member("toSimPlusDelayPeriod").asReal() = oldObj.member("transitionPlusDelayPeriod").asReal();
	}

	static void Update_hclSkinOperator ( hkVariant& oldVar, hkVariant& newVar, hkObjectUpdateTracker& tracker )
	{
		// HCL-127: We've added start and end vertices parameters. For old assets, these should be set to 0 and numVertices-1
		// Number of vertices can be guessed from the number of entries in "boneInfluencesStartPerVertex" in those assets,
		// which contains numVertices+1 entries.

		hkClassAccessor oldObj (oldVar);
		hkClassAccessor newObj (newVar);

		newObj.member("startVertex").asUint32() = 0;
		newObj.member("endVertex").asUint32() = oldObj.member("boneInfluenceStartPerVertex").asSimpleArray().size - 2;
	}

	//HK_COMPILE_TIME_ASSERT(hkSizeOf(DummyMaterial) == hkSizeOf(hkpStorageExtendedMeshShape::Material));
	static void CopyUint32ToMaterial(hkClassMemberAccessor& oldMaterials, hkClassMemberAccessor& newMaterials, hkObjectUpdateTracker& tracker)
	{
		hkClassMemberAccessor::SimpleArray& oldArray = oldMaterials.asSimpleArray();
		DummyArray& newArray = *static_cast<DummyArray*>(newMaterials.getAddress());
		HK_ASSERT(0x650fe9f6, newArray.data == HK_NULL);
		
		initArray(newMaterials, oldArray.size, tracker);

		for( int i = 0; i < oldArray.size; ++i )
		{
			DummyMaterial& material = static_cast<DummyMaterial*>(newArray.data)[i];
			material.filterInfo = static_cast<hkUint32*>(oldArray.data)[i];
			material.restitution = 0.0f;
			material.friction = 1.0f;
			material.userData = 0;
		}
	}

	static void Update_hkpStorageExtendedMeshShapeMeshSubpartStorage(hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker)
	{
		hkClassMemberAccessor oldMaterials(oldObj,"materials");
		hkClassMemberAccessor newMaterials(newObj,"materials");

		CopyUint32ToMaterial(oldMaterials, newMaterials, tracker);
	}

	static void Update_hkpStorageExtendedMeshShapeShapeSubpartStorage(hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker)
	{
		hkClassMemberAccessor oldMaterials(oldObj,"materials");
		hkClassMemberAccessor newMaterials(newObj,"materials");

		CopyUint32ToMaterial(oldMaterials, newMaterials, tracker);
	}

	static void UpdateMaterialStriding( hkClassMemberAccessor& subpart )
	{
		// subpart is an hkArray, so let's use the class accessor to iterate over it
		hkClassMemberAccessor::SimpleArray& subpartArray = subpart.asSimpleArray();
        
		for (int i = 0; i < subpartArray.size; ++i)
		{
			const hkClass* newBufferAccessClass = subpart.getClassMember().getClass();
			HK_ASSERT(0x4858aabb, newBufferAccessClass);
            
            // Index into the array according to its size
			hkVariant newUsedBuffItem = {hkAddByteOffset(subpartArray.data, i * newBufferAccessClass->getObjectSize()), newBufferAccessClass};
            
            // Update the materialStriding to reflect new size of hkpStorageExtendedMeshShape::Material
            hkClassMemberAccessor materialStriding( newUsedBuffItem, "materialStriding");
            materialStriding.asInt16() = sizeof( DummyMaterial );
		}
	}

	static void Update_hkpStorageExtendedMeshShape(hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker)
	{
		// hkpStorageExtendedMeshShape::Material changed size, so we need to version the striding in the arrays of subparts
		hkClassMemberAccessor triSubparts(newObj,"trianglesSubparts");
		UpdateMaterialStriding( triSubparts );

		hkClassMemberAccessor shapesSubparts(newObj,"shapesSubparts");
		UpdateMaterialStriding( shapesSubparts );
	}

	static void Update_hkdSliceFracture(hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker)
	{
		hkClassMemberAccessor oldFracture(oldObj,"childFracture");
		hkClassMemberAccessor newFracture(newObj,"childFracture");
		newFracture.asPointer(0) = oldFracture.asPointer(0);
		tracker.objectPointedBy(newFracture.asPointer(), newFracture.getAddress());
	}

	static void Update_hkdFractureMoveFlattenHierarchy(hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker)
	{
		hkClassMemberAccessor oldMember(oldObj,"flattenHierarchy");
		hkClassMemberAccessor newMember(newObj,"flattenHierarchy");
		newMember.asBool(0) = oldMember.asBool(0);
	}

	static void Update_renamedPropagationRate(hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker)
	{
		hkClassMemberAccessor oldMember(oldObj,"breakingPropogationRate");
		hkClassMemberAccessor newMember(newObj,"breakingPropagationRate");
		newMember.asReal(0) = oldMember.asReal(0);
	}


static const hkVersionRegistry::ClassAction s_updateActions[] =
{
	// variants
	{ 0x6728e4b7, 0x6728e4b7, hkVersionRegistry::VERSION_VARIANT, "hkaBoneAttachment", HK_NULL },
	{ 0xf598a34e, 0xf598a34e, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainer", HK_NULL },
	{ 0x853a899c, 0x853a899c, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainerNamedVariant", HK_NULL },
	{ 0xdce3ca6b, 0xdce3ca6b, hkVersionRegistry::VERSION_VARIANT | hkVersionRegistry::VERSION_COPY, "hkMemoryResourceHandle", HK_NULL },
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

	// common
	{ 0x98e4ebd3, 0x4b2533f6, hkVersionRegistry::VERSION_COPY, "hkSimpleLocalFrame", HK_NULL },

	// physics
	{ 0x2033b565, 0x95b74236, hkVersionRegistry::VERSION_COPY, "hkpConstraintInstance", Update_ConstraintInstance },
	{ 0x959a580d, 0xc92fe7b3, hkVersionRegistry::VERSION_COPY, "hkpEntity", Update_WorldObject }, // Also has changes (in addition to the collidalbleQualityType). Needs its own entry to avoid warnings.
	{ 0x08c08009, 0x6e70ee3f, hkVersionRegistry::VERSION_COPY, "hkpWorldObject", Update_WorldObject }, // hkpCollidable type changed, values shifted, new enum value added
	{ 0x141abdc9, 0xb9256deb, hkVersionRegistry::VERSION_COPY, "hkpMotion", HK_NULL },

	{ 0xa539881b, 0xf4b0f799, hkVersionRegistry::VERSION_COPY, "hkpTypedBroadPhaseHandle", Update_Assert }, // expected to be used as embedded class member only
	{ 0xf204e3b7, 0x1fb16917, hkVersionRegistry::VERSION_COPY, "hkpCollidable", Update_Assert }, // expected to be used as embedded class member only

	{ 0x77102e99, 0x0d18875d, hkVersionRegistry::VERSION_COPY, "hkpStorageExtendedMeshShape", Update_hkpStorageExtendedMeshShape },	
	{ 0x85228673, 0xacbe266e, hkVersionRegistry::VERSION_COPY, "hkpExtendedMeshShape", HK_NULL },
	{ 0x3375309c, 0x36469972, hkVersionRegistry::VERSION_COPY, "hkpExtendedMeshShapeTrianglesSubpart", HK_NULL },
	{ 0x3b2f0b51, 0x1e37d40d, hkVersionRegistry::VERSION_COPY, "hkpStorageExtendedMeshShapeMeshSubpartStorage", Update_hkpStorageExtendedMeshShapeMeshSubpartStorage },
	{ 0x502939ed, 0x14d4585e, hkVersionRegistry::VERSION_COPY, "hkpStorageExtendedMeshShapeShapeSubpartStorage", Update_hkpStorageExtendedMeshShapeShapeSubpartStorage },
	{ 0x7c9435c7, 0x5d634d20, hkVersionRegistry::VERSION_COPY, "hkpExtendedMeshShapeSubpart", HK_NULL },

	{ 0xc89cb00f, 0x4e6c0972, hkVersionRegistry::VERSION_COPY, "hkpWorldCinfo", HK_NULL },  // new members for kd-tree
	
	// behavior
	BINARY_IDENTICAL( 0xad067113, 0x76bddb31, "hkbEvent" ),
	BINARY_IDENTICAL( 0x1f94374a, 0xb8153dd6, "hkbEventSequencedData" ),
	BINARY_IDENTICAL( 0x66316e5a, 0x02d59d6e, "hkbEventSequencedDataSequencedEvent" ),
	BINARY_IDENTICAL( 0x6369055f, 0x00c46650, "hkbStringPredicate" ),
	{ 0x85579f84, 0xf689eb05, hkVersionRegistry::VERSION_COPY, "hkbAttachmentModifier", Update_hkbAttachmentModifier },
	{ 0x15d7d9d8, 0x6d076c86, hkVersionRegistry::VERSION_COPY, "hkbAttachmentSetup", HK_NULL },
	{ 0xd3b3efc7, 0x8b8b1335, hkVersionRegistry::VERSION_COPY, "hkbBehaviorGraph", HK_NULL },
	{ 0x2f0cf6c5, 0x01e05e53, hkVersionRegistry::VERSION_COPY, "hkbBehaviorGraphData", Update_hkbBehaviorGraphData },
	{ 0xedb19818, 0x9c5d4a32, hkVersionRegistry::VERSION_COPY, "hkbBehaviorGraphStringData", HK_NULL },
	{ 0xc99c5164, 0xf9f192ae, hkVersionRegistry::VERSION_COPY, "hkbBlenderGeneratorChild", Update_hkbBlenderGeneratorChild },
	{ 0x7a9e88e0, 0xa27053d5, hkVersionRegistry::VERSION_COPY, "hkbBlendingTransitionEffect", HK_NULL },
	{ 0x2832e9c4, 0x6cfeff3b, hkVersionRegistry::VERSION_COPY, "hkbCatchFallModifier", Update_hkbCatchFallModifier },
	{ 0xa33104be, 0x4b3e5b60, hkVersionRegistry::VERSION_COPY, "hkbCharacterData", HK_NULL },
	{ 0x632ff427, 0x52bd5952, hkVersionRegistry::VERSION_COPY, "hkbCharacterStringData", HK_NULL },
	{ 0x641f10e0, 0xefbf6f70, hkVersionRegistry::VERSION_COPY, "hkbCheckBalanceModifier", HK_NULL },
	{ 0x06fc48fb, 0xbe629a83, hkVersionRegistry::VERSION_COPY, "hkbCheckRagdollSpeedModifier", Update_hkbCheckRagdollSpeedModifier },
	{ 0xd05509cd, 0x9470551d, hkVersionRegistry::VERSION_COPY, "hkbClipGenerator", Update_hkbClipGenerator },
	{ 0x57a677e8, 0x7eb45cea, hkVersionRegistry::VERSION_COPY, "hkbClipTrigger", HK_NULL },
	{ 0x7c0f82f5, 0xcd349ffa, hkVersionRegistry::VERSION_COPY, "hkbDetectCloseToGroundModifier", Update_hkbDetectCloseToGroundModifier },
	{ 0xbff4f102, 0x9f54d0cb, hkVersionRegistry::VERSION_COPY, "hkbEvaluateHandleModifier", HK_NULL },
	{ 0xbd3a7d99, 0xbd8fae87, hkVersionRegistry::VERSION_COPY, "hkbFootIkModifierLeg", HK_NULL },
	{ 0xe2eb75fc, 0xe5ca3677, hkVersionRegistry::VERSION_COPY, "hkbFootIkModifierInternalLegData", HK_NULL },
	{ 0xfb03b559, 0x496801c9, hkVersionRegistry::VERSION_COPY, "hkbFootIkModifier", Update_hkbFootIkModifier },
	{ 0xf3dec4db, 0xb0ea3e7a, hkVersionRegistry::VERSION_COPY, "hkbGeneratorTransitionEffect", HK_NULL },
	{ 0xe08cfea7, 0x71c065ca, hkVersionRegistry::VERSION_COPY, "hkbHandIkControlData", HK_NULL },
	{ 0xd87204b0, 0x55ef32bf, hkVersionRegistry::VERSION_COPY, "hkbHandIkControlsModifier", HK_NULL }, 
	{ 0x5ba5955a, 0x3c336775, hkVersionRegistry::VERSION_COPY, "hkbHandIkControlsModifierHand", HK_NULL },
	{ 0xa054e9e1, 0xb01b817f, hkVersionRegistry::VERSION_COPY, "hkbHandIkModifier", HK_NULL },
	{ 0xe5396080, 0xe12f69f1, hkVersionRegistry::VERSION_COPY, "hkbHandIkModifierHand", HK_NULL },
	{ 0xda4c7e80, 0xccd695a7, hkVersionRegistry::VERSION_COPY, "hkbJigglerGroup", Update_hkbJigglerGroup },
	{ 0x34e2ace2, 0x9aa4ff71, hkVersionRegistry::VERSION_COPY, "hkbKeyframeBonesModifier", Update_hkbKeyframeBonesModifier },
	{ 0xf3783030, 0xd39a83be, hkVersionRegistry::VERSION_COPY, "hkbLookAtModifier", HK_NULL },
	{ 0x9889e274, 0xa99bbc69, hkVersionRegistry::VERSION_COPY, "hkbMoveBoneTowardTargetModifier", Update_hkbMoveBoneTowardTargetModifier },
	{ 0x50ed39ad, 0xdf07c1cf, hkVersionRegistry::VERSION_COPY, "hkbPositionRelativeSelectorGenerator", Update_hkbPositionRelativeSelectorGenerator },
	{ 0x4a47ab5b, 0x36a118b8, hkVersionRegistry::VERSION_COPY, "hkbPoweredRagdollControlsModifier", Update_hkbPoweredRagdollControlsModifier },
	{ 0xb69595c4, 0xd06d972f, hkVersionRegistry::VERSION_COPY, "hkbPoweredRagdollModifier", HK_NULL },
	{ 0x7ba0b307, 0xfff770ad, hkVersionRegistry::VERSION_COPY, "hkbRegisteredGenerator", HK_NULL },
	{ 0x3c2ebb92, 0xdd8f876a, hkVersionRegistry::VERSION_COPY, "hkbRigidBodyRagdollModifier", Update_hkbRigidBodyRagdollModifier },
	{ 0x22d96c70, 0x1e9bb777, hkVersionRegistry::VERSION_COPY, "hkbSenseHandleModifier", HK_NULL },
	{ 0xcde98636, 0x449e8371, hkVersionRegistry::VERSION_COPY, "hkbSimpleCharacter", HK_NULL },
	{ 0x267e83fb, 0x9f72e732, hkVersionRegistry::VERSION_COPY, "hkbSplinePathGenerator", Update_hkbSplinePathGenerator },
	{ 0x0f7b063e, 0x68966be4, hkVersionRegistry::VERSION_COPY, "hkbStateMachineStateInfo", Update_hkbStateMachineStateInfo },
	{ 0x0718b7a9, 0xcdec8025, hkVersionRegistry::VERSION_COPY, "hkbStateMachineTransitionInfo", Update_hkbStateMachineTransitionInfo },
	{ 0xbcfd33eb, 0xe94cc5bc, hkVersionRegistry::VERSION_COPY, "hkbStateMachine", Update_hkbStateMachine },
	{ 0xba7763cb, 0x91f90c22, hkVersionRegistry::VERSION_COPY, "hkbStateMachineProspectiveTransitionInfo", HK_NULL },
	{ 0x8f9ec696, 0x2f375780, hkVersionRegistry::VERSION_COPY, "hkbTargetRigidBodyModifier", Update_hkbTargetRigidBodyModifier },
	{ 0xd018c562, 0x737d9c51, hkVersionRegistry::VERSION_COPY, "hkbTimerModifier", Update_hkbTimerModifier },
	{ 0x21898fef, 0x6a6beab7, hkVersionRegistry::VERSION_COPY, "hkbVariableBindingSet", Update_hkbVariableBindingSet },
	{ 0x0487a360, 0x791fb0b1, hkVersionRegistry::VERSION_COPY, "hkbVariableBindingSetBinding", HK_NULL },
	{ 0x0cf63a99, 0x9e746ba2, hkVersionRegistry::VERSION_COPY, "hkbVariableInfo", HK_NULL },
	{ 0x2de24779, 0xcc088ba1, hkVersionRegistry::VERSION_COPY, "hkbNode", HK_NULL },

	// destruction
	{ 0xc62ede40, 0xbe975804, hkVersionRegistry::VERSION_COPY, "hkdBreakableShape", Update_renamedPropagationRate },
	{ 0x8855fc1e, 0x5bdd6792, hkVersionRegistry::VERSION_COPY, "hkdController", HK_NULL },
	{ 0x50ae5c9e, 0x36d49c44, hkVersionRegistry::VERSION_COPY, "hkdSliceFracture", Update_hkdSliceFracture },
	{ 0x129e843a, 0xce2b035d, hkVersionRegistry::VERSION_COPY, "hkdWoodFracture", Update_hkdFractureMoveFlattenHierarchy },
	{ 0xa36460de, 0xf01bd406, hkVersionRegistry::VERSION_COPY, "hkdSplitInHalfFracture", Update_hkdFractureMoveFlattenHierarchy },
	{ 0xa55c2a5a, 0x8e16744d, hkVersionRegistry::VERSION_COPY, "hkdFracture", HK_NULL },
	{ 0x55182bda, 0xf7925275, hkVersionRegistry::VERSION_COPY, "hkdBody", HK_NULL },
	{ 0xf3b2e806, 0x74601df3, hkVersionRegistry::VERSION_COPY, "hkdBreakableBodyBlueprint", HK_NULL },
	{ 0x2468f054, 0xa71c409c, hkVersionRegistry::VERSION_COPY|hkVersionRegistry::VERSION_VARIANT, "hkdShape", Update_renamedPropagationRate },
	REMOVED("hkdCompoundBreakableBodyBlueprint"),
	REMOVED("hkdBallGunBlueprint"), // HKD-227
	REMOVED("hkdChangeMassGunBlueprint"),
	REMOVED("hkdGravityGunBlueprint"),
	REMOVED("hkdGrenadeGunBlueprint"),
	REMOVED("hkdWeaponBlueprint"),

	// cloth
	{ 0xd6ac84c1, 0x0709cc57, hkVersionRegistry::VERSION_COPY, "hclTransitionConstraintSetPerParticle", Update_hclTransitionConstraintSetPerParticle },
	{ 0x8b20cab6, 0xc1a80a0b, hkVersionRegistry::VERSION_COPY, "hclTransitionConstraintSet", Update_hclTransitionConstraintSet },
	{ 0xb4f4c452, 0xd10ab1c8, hkVersionRegistry::VERSION_COPY, "hclSkinOperator", Update_hclSkinOperator },

	
	{ 0, 0, 0, HK_NULL, HK_NULL }
};	 

static const hkVersionRegistry::ClassRename s_renames[] =
{
	{ "hkbPredicate", "hkbCondition" },
	{ "hkbStringPredicate", "hkbStringCondition" },
	{ "hkbClimbMountingPredicate", "hkbClimbMountingCondition" },
	{ HK_NULL, HK_NULL }
};

#define HK_COMPAT_VERSION_FROM hkHavok610r1Classes
#define HK_COMPAT_VERSION_TO hkHavok650b1Classes
#define HK_COMPAT_OPTIONAL_UPDATE_FUNC update

extern hkVersionRegistry::UpdateDescription hkVersionUpdateDescription;

static hkResult HK_CALL update(
							   hkArray<hkVariant>& objectsInOut,
							   hkObjectUpdateTracker& tracker )
{
	Context context(objectsInOut, tracker);

	context.putVariableBindingSetObjectsLast();

	return hkVersionUtil::updateSingleVersion( objectsInOut, tracker, hkVersionUpdateDescription, hkVersionRegistry::getInstance().getClassNameRegistry(HK_COMPAT_VERSION_TO::VersionString) );
}

#include<Common/Compat/Deprecated/Compat/hkCompat_Common.cxx>
#undef HK_COMPAT_VERSION_FROM
#undef HK_COMPAT_VERSION_TO

} // namespace hkCompat_hk610r1_hk650b1

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
