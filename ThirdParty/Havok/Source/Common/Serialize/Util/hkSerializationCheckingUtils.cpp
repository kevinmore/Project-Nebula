/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Util/hkSerializationCheckingUtils.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Reflection/hkInternalClassMember.h>
#include <Common/Base/Reflection/Registry/hkClassNameRegistry.h>
#include <Common/Base/Reflection/Registry/hkTypeInfoRegistry.h>
#include <Common/Base/Reflection/Registry/hkVtableClassRegistry.h>
#include <Common/Serialize/Util/hkBuiltinTypeRegistry.h>

namespace
{
	class DataTestController : public hkReferencedObject
	{
		public:
			virtual void set(void* address, const hkClass* klass, const hkClassMember* member, const hkTypeInfoRegistry& typeInfoReg) = 0;
			virtual hkResult process(void* address, const hkClass* klass, const hkClassMember* member, const hkTypeInfoRegistry& typeInfoReg, hkOstream& report) = 0;

		protected:
			typedef hkPointerMap<void*, void*> AllocationFromLocationMap;
	};

#define HK_REFLECTION_VERIFY_DATA 0xf0
#define HK_REFLECTION_VERIFY_DEFAULT_REF_COUNT 100

	HK_COMPILE_TIME_ASSERT(hkSizeOf(hkStringPtr) == hkSizeOf(char*));
	HK_ALIGN( const char StringPtrTestController_s_testString[], 4) = "This is a test string";
	
	class StringPtrTestController : public DataTestController
	{
		public:
			virtual void set(void* address, const hkClass* klass, const hkClassMember* member, const hkTypeInfoRegistry& typeInfoReg)
			{
				*static_cast<const char**>(address) = StringPtrTestController_s_testString;
			}

			virtual hkResult process(void* address, const hkClass* klass, const hkClassMember* member, const hkTypeInfoRegistry& typeInfoReg, hkOstream& report)
			{
				hkStringPtr& strPtr = *static_cast<hkStringPtr*>(address);
				if( strPtr.cString() == StringPtrTestController_s_testString && (*static_cast<hkUlong*>(address) & hkStringPtr::OWNED_FLAG) == 0 )
				{
					return HK_SUCCESS;
				}
				return HK_FAILURE;
			}
	};


	class ArrayTestController : public DataTestController
	{
		public:

			struct DummyArray
			{
				void* data;
				int size;
				int capacity;
			};

			ArrayTestController() : m_defaultSize(5) {}

			~ArrayTestController()
			{
				for( int i = 0; i < m_allocations.getSize(); ++i )
				{
					hkDeallocateChunk<char>(static_cast<char*>(m_allocations[i].data), m_allocations[i].size, HK_MEMORY_CLASS_ARRAY);
				}
			}

			virtual void set(void* address, const hkClass* klass, const hkClassMember* member, const hkTypeInfoRegistry& typeInfoReg)
			{
				HK_ASSERT(0x41407a4f, klass && member);
				if( member->getSubType() == hkClassMember::TYPE_VOID )
				{
					return;
				}
				DummyArray* arr = static_cast<DummyArray*>(address);
				arr->size = m_defaultSize;
				int dataSize = member->getArrayMemberSize()*arr->size;
				arr->data = hkAllocateChunk<char>(dataSize, HK_MEMORY_CLASS_ARRAY);
				hkString::memSet(arr->data, 0, dataSize);
				trackAllocation(address, arr->data, dataSize);
				if( member->getType() == hkClassMember::TYPE_ARRAY )
				{
					arr->capacity = m_defaultSize;
				}
			}

			virtual hkResult process(void* address, const hkClass* klass, const hkClassMember* member, const hkTypeInfoRegistry& typeInfoReg, hkOstream& report)
			{
				HK_ASSERT(0x41407a50, klass && member);
				if( member->getSubType() == hkClassMember::TYPE_VOID )
				{
					return HK_SUCCESS;
				}
				HK_ASSERT(0x41407a51, m_allocationFromLocation.hasKey(address) && m_allocationFromLocation.getWithDefault(address, HK_NULL));
				DummyArray* arr = static_cast<DummyArray*>(address);
				if( member->getType() == hkClassMember::TYPE_ARRAY )
				{
					if( arr->size == 0 && arr->capacity == hkArray<char>::DONT_DEALLOCATE_FLAG && arr->data == HK_NULL )
					{
						report.printf("%s class finish constructor does not call m_%s member's finish constructor.\n", klass->getName(), member->getName());
						return HK_FAILURE;
					}
				}
				if( arr->size != m_defaultSize
					|| arr->data != m_allocationFromLocation.getWithDefault(address, HK_NULL)
					|| (member->getType() == hkClassMember::TYPE_ARRAY && arr->capacity != m_defaultSize) )
				{
					report.printf("%s class has its m_%s member overwritten.\n", klass->getName(), member->getName());
					return HK_FAILURE;
				}
				return HK_SUCCESS;
			}

		protected:

			void trackAllocation(void* address, void* data, int size)
			{
				AllocationInfo& info = m_allocations.expandOne();
				info.data = data;
				info.size = size;
				m_allocationFromLocation.insert(address, data);
			}

		protected:

			hkInt32 m_defaultSize;
			struct AllocationInfo
			{
				void* data;
				int size;
			};
			hkArray<AllocationInfo> m_allocations;
			AllocationFromLocationMap m_allocationFromLocation;
	};

	struct TestStruct
	{
		HK_DECLARE_REFLECTION();
		hkInt32 m_value;
	};
	const hkInternalClassMember TestStruct::Members[] =
	{
		{ "value", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(TestStruct,m_value), HK_NULL }
	};
	const hkClass TestStructClass(
		"TestStruct",
		HK_NULL, // parent
		sizeof(TestStruct),
		HK_NULL,
		0, // interfaces
		HK_NULL,
		0, // enums
		reinterpret_cast<const hkClassMember*>(TestStruct::Members),
		HK_COUNT_OF(TestStruct::Members),
		HK_NULL, // defaults
		HK_NULL, // attributes
		0, // flags
		0 // version
		);

	class HomogeneousArrayTestController : public ArrayTestController
	{
		public:

			struct HomogeneousArray
			{
				const hkClass* klass;
				void* data;
				int size;
			};

			virtual void set(void* address, const hkClass* klass, const hkClassMember* member, const hkTypeInfoRegistry& typeInfoReg)
			{
				HK_ASSERT(0x41407a4f, klass && member);
				HomogeneousArray* arr = static_cast<HomogeneousArray*>(address);
				arr->klass = &TestStructClass;
				arr->size = m_defaultSize;
				int dataSize = hkSizeOf(TestStruct)*arr->size;
				arr->data = hkAllocateChunk<char>(dataSize, HK_MEMORY_CLASS_ARRAY);
				hkString::memSet(arr->data, 0, dataSize);
				trackAllocation(address, arr->data, dataSize);
			}

			virtual hkResult process(void* address, const hkClass* klass, const hkClassMember* member, const hkTypeInfoRegistry& typeInfoReg, hkOstream& report)
			{
				HK_ASSERT(0x41407a50, klass && member);
				HK_ASSERT(0x41407a51, m_allocationFromLocation.hasKey(address) && m_allocationFromLocation.getWithDefault(address, HK_NULL));
				HomogeneousArray* arr = static_cast<HomogeneousArray*>(address);
				if( arr->klass != &TestStructClass
					|| arr->size != m_defaultSize
					|| arr->data != m_allocationFromLocation.getWithDefault(address, HK_NULL) )
				{
					report.printf("%s class has its m_%s member overwritten.\n", klass->getName(), member->getName());
					return HK_FAILURE;
				}
				return HK_SUCCESS;
			}
	};

	class EmbeddedObjectTestController;

#	define EMBEDDED_MEMBER_TO_TEST reinterpret_cast<const hkClassMember*>(-1)

	class ObjectTestController : public DataTestController
	{
		public:
			typedef hkPointerMap<const hkClassMember*, int> FlagFromClassMemberMap;
			static FlagFromClassMemberMap& getFlagFromClassMemberMap()
			{
				if( s_processFlagFromClassMember == HK_NULL )
				{
					s_processFlagFromClassMember = new FlagFromClassMemberMap();
				}
				return *s_processFlagFromClassMember;
			}
			static void clearFlagFromClassMemberMap()
			{
				if( s_processFlagFromClassMember )
				{
					delete s_processFlagFromClassMember;
					s_processFlagFromClassMember = HK_NULL;
				}
			}

		private:
			static hkPointerMap<const hkClassMember*, int>* s_processFlagFromClassMember;

		public:
			typedef hkPointerMap<void*, DataTestController*> ObjectTestFromLocationMap;
			typedef hkPointerMap<hkReferencedObject*, int> ReferencedObjectAllocationsMap;

			ObjectTestController() {}
			~ObjectTestController()
			{
				HK_ASSERT(0x41407a52, m_objectTestFromLocationMap.getSize() == 0);
				for( ReferencedObjectAllocationsMap::Iterator iter = m_refCountFromObjectMap.getIterator(); m_refCountFromObjectMap.isValid(iter); iter = m_refCountFromObjectMap.getNext(iter) )
				{
					hkReferencedObject* p = m_refCountFromObjectMap.getKey(iter);
					delete p;
				}
			}

			virtual void set(void* address, const hkClass* klass, const hkClassMember* member, const hkTypeInfoRegistry& typeInfoReg)
			{
				HK_ASSERT(0x41407a53, klass);
				HK_ASSERT(0x41407a54, member == HK_NULL);

				if( hkReferencedObjectClass.isSuperClass(*klass) )
				{
					setReferencedObject(address);
				}

				for( int i = 0; i < klass->getNumMembers(); ++i )
				{
					const hkClassMember& structMember = klass->getMember(i);
					if( structMember.getFlags().allAreSet(hkClassMember::SERIALIZE_IGNORED) )
					{
						continue;
					}
					void* memaddr = hkAddByteOffset(address, structMember.getOffset());
					if( !setSimpleType(memaddr, structMember) )
					{
						if( !setComplexType(memaddr, *klass, structMember, typeInfoReg) )
						{
							HK_ASSERT2(0x5749b988, false, "Can not test class member of unknown type.");
						}
					}
				}
			}

			virtual hkResult postprocess(void* address, const hkClass* klass, const hkClassMember* member, const hkTypeInfoRegistry& typeInfoReg, hkOstream& report)
			{
				const char* className = klass->getName();
				HK_ASSERT(0x41407a56, klass);
				HK_ASSERT(0x41407a57, member == HK_NULL || member == EMBEDDED_MEMBER_TO_TEST);
				for( ReferencedObjectAllocationsMap::Iterator iter = m_refCountFromObjectMap.getIterator(); m_refCountFromObjectMap.isValid(iter); iter = m_refCountFromObjectMap.getNext(iter) )
				{
					hkReferencedObject* p = m_refCountFromObjectMap.getKey(iter);
					int refCount = m_refCountFromObjectMap.getValue(iter);
					if( p->getReferenceCount() == refCount )
					{
						report.printf("%s class destructor is not handling reference counted objects.\n", className);
					}
				}
				return HK_SUCCESS;
			}

			virtual hkResult process(void* address, const hkClass* klass, const hkClassMember* member, const hkTypeInfoRegistry& typeInfoReg, hkOstream& report)
			{
				HK_ASSERT(0x41407a56, klass);
				HK_ASSERT(0x41407a57, member == HK_NULL || member == EMBEDDED_MEMBER_TO_TEST);
				hkResult res = HK_SUCCESS;
				const char* className = klass->getName();
				const hkTypeInfo* typeInfo = typeInfoReg.getTypeInfo( className );
				HK_ASSERT(0x41407a58, typeInfo);
				hkBool classHasFinishCtor = typeInfo->hasFinishFunction();

				if( klass->hasVtable() && !classHasFinishCtor )
				{
					report.printf("%s class is virtual, but no finish constructor is defined.\n", className);
					return HK_FAILURE;
				}

				{
					if( hkReferencedObjectClass.isSuperClass(*klass) && processReferencedObject(address) == HK_FAILURE )
					{
						res = HK_FAILURE;
						report.printf("%s class finish constructor does not call parent's finish constructor, hkReferencedObject finish constructor is not called.\n", className);
					}
					for( int i = 0; i < klass->getNumMembers(); ++i )
					{
						const hkClassMember& classMember = klass->getMember(i);
						if( classMember.getFlags().allAreSet(hkClassMember::SERIALIZE_IGNORED) )
						{
							continue;
						}
						hkResult memberResult = HK_SUCCESS;
						const char* memberName = classMember.getName();
						void* memaddr = hkAddByteOffset(address, classMember.getOffset());
						hkBool32 memberHasFinishCtor = false;

						if( processSimpleType(memaddr, classMember, memberResult) )
						{
							if( memberResult == HK_FAILURE )
							{
								if( classHasFinishCtor )
								{
									if( member != EMBEDDED_MEMBER_TO_TEST )
									{
										report.printf("%s class has its m_%s member overwritten. Make sure the class finish constructor checks for finishing flag and calls parent's finish constructor.\n", className, memberName);
									}
								}
								else
								{
									report.printf("%s class has no finish constructor, m_%s member is overwritten.\n", className, memberName);
								}
								res = HK_FAILURE;
							}
							else if( memberHasFinishCtor && !classHasFinishCtor )
							{
								report.printf("%s class has no finish constructor, but its m_%s member requires it.\n", className, memberName);
								res = HK_FAILURE;
							}
						}
						else if( processComplexType(memaddr, *klass, classMember, typeInfoReg, memberHasFinishCtor, memberResult, report) )
						{
							if( memberResult == HK_FAILURE )
							{
								if( memberHasFinishCtor )
								{
									if( classHasFinishCtor )
									{
										if( member != EMBEDDED_MEMBER_TO_TEST )
										{
											if( classMember.hasClass() )
											{
												report.printf("%s class has its m_%s member overwritten. Make sure the class finish constructor checks for finishing flag and calls parent's finish constructor.\n", className, memberName);
											}
											else
											{
												report.printf("%s class has its m_%s member overwritten. Make sure you call member's finish constructor.\n", className, memberName);
											}
										}
									}
									else
									{
										report.printf("%s class has m_%s member with finish constructor, but no class finish constructor defined.\n", className, memberName);
									}
								}
								else
								{
									report.printf("%s class has its m_%s member overwritten.\n", className, memberName);
								}
								res = HK_FAILURE;
							}
							else if( memberHasFinishCtor && !classHasFinishCtor )
							{
								report.printf("%s class has no finish constructor, but its m_%s member requires it.\n", className, memberName);
								res = HK_FAILURE;
							}
						}
						else
						{
							HK_ASSERT2(0x12530f98, false, "Cannot test class member of unknown type.");
						}
					}
				}
				return res;
			}

		protected:

			ObjectTestFromLocationMap m_objectTestFromLocationMap;
			ReferencedObjectAllocationsMap m_refCountFromObjectMap;

			hkBool32 setSimpleType(void* memaddr, const hkClassMember& member);
			hkBool32 setComplexType(void* memaddr, const hkClass& klass, const hkClassMember& member, const hkTypeInfoRegistry& typeInfoReg);
			hkBool32 processSimpleType(void* memaddr, const hkClassMember& member, hkResult& result);
			hkBool32 processComplexType(void* memaddr, const hkClass& klass, const hkClassMember& member, const hkTypeInfoRegistry& typeInfoReg, hkBool32& memberHasFinishCtor, hkResult& result, hkOstream& report);

			void setReferencedObject(void* memaddr)
			{
				HK_COMPILE_TIME_ASSERT(hkSizeOf(static_cast<hkReferencedObject*>(memaddr)->m_memSizeAndFlags) == hkSizeOf(hkUint16));
				HK_COMPILE_TIME_ASSERT(hkSizeOf(static_cast<hkReferencedObject*>(memaddr)->m_referenceCount) == hkSizeOf(hkUint16));

				HK_ASSERT(0x41407a5c, static_cast<hkReferencedObject*>(memaddr)->m_memSizeAndFlags == 0);
				hkString::memSet(&static_cast<hkReferencedObject*>(memaddr)->m_memSizeAndFlags, HK_REFLECTION_VERIFY_DATA, hkSizeOf(hkUint16));
				HK_ASSERT(0x41407a5c, static_cast<hkReferencedObject*>(memaddr)->m_referenceCount == 0);
				hkString::memSet(&static_cast<hkReferencedObject*>(memaddr)->m_referenceCount, HK_REFLECTION_VERIFY_DATA, hkSizeOf(hkUint16));
			}

			hkResult processReferencedObject(void* memaddr)
			{
				hkUint16 testValue;
				hkString::memSet(&testValue, HK_REFLECTION_VERIFY_DATA, hkSizeOf(hkUint16));
				return hkString::memCmp(&static_cast<hkReferencedObject*>(memaddr)->m_memSizeAndFlags, &testValue, hkSizeOf(hkUint16)) == 0
						&& hkString::memCmp(&static_cast<hkReferencedObject*>(memaddr)->m_referenceCount, &testValue, hkSizeOf(hkUint16)) == 0 ? HK_SUCCESS : HK_FAILURE;
			}
	};
	ObjectTestController::FlagFromClassMemberMap* ObjectTestController::s_processFlagFromClassMember;

	class EmbeddedObjectTestController : public ObjectTestController
	{
		public:

			EmbeddedObjectTestController() {}

			virtual void set(void* memaddr, const hkClass* ownerClass, const hkClassMember* structMember, const hkTypeInfoRegistry& typeInfoReg)
			{
				HK_ASSERT(0x41407a59, ownerClass && structMember && structMember->hasClass());
				const hkClass& structClass = structMember->getStructClass();

				ObjectTestController::set(memaddr, &structClass, HK_NULL, typeInfoReg);
			}

			virtual hkResult process(void* memaddr, const hkClass* ownerClass, const hkClassMember* structMember, const hkTypeInfoRegistry& typeInfoReg, hkOstream& report)
			{
				HK_ASSERT(0x41407a5b, ownerClass && structMember && structMember->hasClass());
				hkResult res = HK_SUCCESS;
				const char* ownerClassName = ownerClass->getName();
				const hkClass& structClass = structMember->getStructClass();

				if( structClass.hasVtable() && !typeInfoReg.getTypeInfo( ownerClassName )->hasFinishFunction() )
				{
					res = HK_FAILURE;
					report.printf("%s class has virtual member m_%s, but no class finish constructor defined.\n", ownerClassName, structMember->getName() );
				}
				if( ObjectTestController::process(memaddr, &structClass, EMBEDDED_MEMBER_TO_TEST, typeInfoReg, report) == HK_FAILURE )
				{
					return HK_FAILURE;
				}
				return res;
			}
	};

	hkBool32 ObjectTestController::setSimpleType(void* memaddr, const hkClassMember& member)
	{
		if( member.getType() != hkClassMember::TYPE_STRUCT
			&& member.getType() != hkClassMember::TYPE_ARRAY
			&& member.getType() != hkClassMember::TYPE_SIMPLEARRAY
			&& member.getType() != hkClassMember::TYPE_HOMOGENEOUSARRAY
			&& member.getType() != hkClassMember::TYPE_STRINGPTR )
		{
			HK_ASSERT(0x41407a5d, *static_cast<char*>(memaddr) == 0);
			if( member.getType() == hkClassMember::TYPE_POINTER && member.getSubType() == hkClassMember::TYPE_STRUCT
				&& member.hasClass() && hkReferencedObjectClass.isSuperClass(member.getStructClass()) )
			{
				int cArraySize = member.getCstyleArraySize() == 0 ? 1 : member.getCstyleArraySize();
				HK_ASSERT(0x1681ff5e, member.getSizeInBytes() == cArraySize*hkSizeOf(hkReferencedObject*));
				for( int i = 0; i < cArraySize; ++i )
				{
					hkReferencedObject* p = new hkReferencedObject();
					p->m_referenceCount = HK_REFLECTION_VERIFY_DEFAULT_REF_COUNT;
					m_refCountFromObjectMap.insert(p, p->m_referenceCount);
					hkString::memCpy(&static_cast<hkReferencedObject**>(memaddr)[i], &p, sizeof(hkReferencedObject*));
				}
			}
			else
			{
				hkString::memSet(memaddr, HK_REFLECTION_VERIFY_DATA, member.getSizeInBytes());
			}
			return true;
		}
		return false;
	}

	hkBool32 ObjectTestController::setComplexType(void* memaddr, const hkClass& klass, const hkClassMember& member, const hkTypeInfoRegistry& typeInfoReg)
	{
		HK_ASSERT(0x41407a5e, *static_cast<char*>(memaddr) == 0);
		HK_ASSERT(0x41407a5f, m_objectTestFromLocationMap.hasKey(memaddr) == false);
		switch( member.getType() )
		{
			case hkClassMember::TYPE_STRINGPTR:
			{
				HK_ASSERT(0x27742d4c, m_objectTestFromLocationMap.hasKey(memaddr) == false);
				StringPtrTestController* testString = new StringPtrTestController();
				m_objectTestFromLocationMap.insert(memaddr, testString);
				testString->set(memaddr, &klass, &member, typeInfoReg);
				return true;
			}
			case hkClassMember::TYPE_STRUCT:
			{
				HK_ASSERT(0x41407a60, m_objectTestFromLocationMap.hasKey(memaddr) == false);
				EmbeddedObjectTestController* testStruct = new EmbeddedObjectTestController();
				m_objectTestFromLocationMap.insert(memaddr, testStruct);
				testStruct->set(memaddr, &klass, &member, typeInfoReg);
				return true;
			}
			case hkClassMember::TYPE_ARRAY:
			case hkClassMember::TYPE_SIMPLEARRAY:
			{
				HK_ASSERT(0x41407a61, m_objectTestFromLocationMap.hasKey(memaddr) == false);
				ArrayTestController* testArray = new ArrayTestController();
				m_objectTestFromLocationMap.insert(memaddr, testArray);
				testArray->set(memaddr, &klass, &member, typeInfoReg);
				return true;
			}
			case hkClassMember::TYPE_HOMOGENEOUSARRAY:
			{
				HK_ASSERT(0x41407a67, m_objectTestFromLocationMap.hasKey(memaddr) == false);
				HomogeneousArrayTestController* testArray = new HomogeneousArrayTestController();
				m_objectTestFromLocationMap.insert(memaddr, testArray);
				testArray->set(memaddr, &klass, &member, typeInfoReg);
				return true;
			}
			default:
			{
				HK_ASSERT2(0x41407a62, false, "Can not test class member of unknown type.");
				break;
			}
		}
		return false;
	}

	hkBool32 ObjectTestController::processSimpleType(void* memaddr, const hkClassMember& member, hkResult& result)
	{
		if( member.getType() != hkClassMember::TYPE_STRUCT
			&& member.getType() != hkClassMember::TYPE_ARRAY
			&& member.getType() != hkClassMember::TYPE_SIMPLEARRAY
			&& member.getType() != hkClassMember::TYPE_HOMOGENEOUSARRAY
			&& member.getType() != hkClassMember::TYPE_STRINGPTR )
		{
			if( member.getType() == hkClassMember::TYPE_POINTER && member.getSubType() == hkClassMember::TYPE_STRUCT
				&& member.hasClass() && hkReferencedObjectClass.isSuperClass(member.getStructClass()) )
			{
				int cArraySize = member.getCstyleArraySize() == 0 ? 1 : member.getCstyleArraySize();
				HK_ASSERT(0x26569681, member.getSizeInBytes() == cArraySize*hkSizeOf(hkReferencedObject*));
				for( int i = 0; i < cArraySize; ++i )
				{
					hkReferencedObject* p = static_cast<hkReferencedObject**>(memaddr)[i];
					if( m_refCountFromObjectMap.getWithDefault(p, 0) )
					{
						result = p->getReferenceCount() == HK_REFLECTION_VERIFY_DEFAULT_REF_COUNT ? HK_SUCCESS : HK_FAILURE;
					}
					else
					{
						result = HK_FAILURE;
					}
					if( result == HK_FAILURE )
					{
						break;
					}
				}
			}
			else
			{
				int size = member.getSizeInBytes();
				hkLocalArray<char> v(size);
				hkString::memSet(v.begin(), HK_REFLECTION_VERIFY_DATA, size);
				result = (hkString::memCmp(v.begin(), memaddr, size) == 0) ? HK_SUCCESS : HK_FAILURE;
			}
			return true;
		}
		return false;
	}

	hkBool32 ObjectTestController::processComplexType(void* memaddr, const hkClass& klass, const hkClassMember& member, const hkTypeInfoRegistry& typeInfoReg, hkBool32& memberHasFinishCtor, hkResult& result, hkOstream& report)
	{
		switch( member.getType() )
		{
			case hkClassMember::TYPE_STRINGPTR:
			{
				DataTestController* testString = m_objectTestFromLocationMap.getWithDefault(memaddr, HK_NULL);
				HK_ASSERT(0x41407a63, testString);
				result = testString->process(memaddr, &klass, &member, typeInfoReg, report);
				m_objectTestFromLocationMap.remove(memaddr);
				testString->removeReference();
				memberHasFinishCtor = true;
				return true;
			}
			case hkClassMember::TYPE_STRUCT:
			{
				DataTestController* testStruct = m_objectTestFromLocationMap.getWithDefault(memaddr, HK_NULL);
				HK_ASSERT(0x41407a63, testStruct);
				result = testStruct->process(memaddr, &klass, &member, typeInfoReg, report);
				m_objectTestFromLocationMap.remove(memaddr);
				testStruct->removeReference();
				memberHasFinishCtor = typeInfoReg.getTypeInfo(member.getStructClass().getName())->hasFinishFunction();
				return true;
			}
			case hkClassMember::TYPE_ARRAY:
			case hkClassMember::TYPE_SIMPLEARRAY:
			case hkClassMember::TYPE_HOMOGENEOUSARRAY:
			{
				DataTestController* testArray = m_objectTestFromLocationMap.getWithDefault(memaddr, HK_NULL);
				HK_ASSERT(0x41407a64, testArray);
				result = testArray->process(memaddr, &klass, &member, typeInfoReg, report);
				m_objectTestFromLocationMap.remove(memaddr);
				testArray->removeReference();
				memberHasFinishCtor = member.getType() == hkClassMember::TYPE_ARRAY;
				return true;
			}
			default:
			{
				HK_ASSERT2(0x41407a65, false, "Can not test class member of unknown type.");
				break;
			}
		}
		return false;
	}
}

static void verifyReflection(const hkClass& klass, const hkTypeInfoRegistry& typeInfoReg, const hkStringMap<int>& vtableFlagFromClassName, hkStringMap<int>& classDoneFromClassName, hkResult& result, hkOstream& report, const char** memoryManagedPrefixes, int numPrefixes, bool reportNonMemoryManaged )
{
	const char* klassName = klass.getName();

	if( classDoneFromClassName.hasKey(klassName) )
	{
		result = (hkResultEnum)classDoneFromClassName.getWithDefault(klassName, HK_FAILURE );
		return;
	}
	result = HK_SUCCESS;
	classDoneFromClassName.insert(klassName, HK_SUCCESS);

	const hkTypeInfo* klassTypeInfo = typeInfoReg.getTypeInfo(klassName);
	bool mayBeAbstract = klassTypeInfo == HK_NULL && klass.getNumInterfaces() > 0;

	int vtable = vtableFlagFromClassName.getWithDefault(klassName, -1);
	if( vtable != -1 ) // vtable is registered
	{
		if( klass.getNumInterfaces() == 0 )
		{
			report.printf("%s class is not virtual, but vtable registry found.\n", klassName );
			result = HK_FAILURE;
		}
		else if( !klassTypeInfo )
		{
			report.printf("%s class is virtual and has no type info. Have you registered the class?\n", klassName );
			result = HK_FAILURE;
		}
		else if( !klassTypeInfo->hasFinishFunction() )
		{
			report.printf("%s class is virtual, but has no finish constructor defined.\n", klassName );
			result = HK_FAILURE;
		}
	}
	else
	{
		if( !mayBeAbstract )
		{
			if( klass.getNumInterfaces() > 0 )
			{
				report.printf("%s class is virtual, but no vtable registry found.\n", klassName );
				result = HK_FAILURE;
			}
			else if( !klassTypeInfo )
			{
				report.printf("%s class is non-virtual and has no type info. Have you registered the class?\n", klassName );
				result = HK_FAILURE;
			}
		}
	}
	if( klass.getParent() )
	{
		verifyReflection(*klass.getParent(), typeInfoReg, vtableFlagFromClassName, classDoneFromClassName, result, report, memoryManagedPrefixes, numPrefixes, reportNonMemoryManaged );
	}
	if( klass.getFlags().allAreSet(hkClass::FLAGS_NOT_SERIALIZABLE) )
	{
		classDoneFromClassName.insert(klassName, result == HK_SUCCESS ? HK_SUCCESS : HK_FAILURE );
		return;
	}
	if( mayBeAbstract && !hkReferencedObjectClass.isSuperClass(klass) && klass.getNumDeclaredMembers() > 0 )
	{
		report.printf("%s class is an interface class, but has members.\n", klassName );
		result = HK_FAILURE;
	}
	// klass is serializable, so check the class members
	{
		hkResult klassMemberResult = HK_SUCCESS;
		for( int i = 0; i < klass.getNumDeclaredMembers(); ++i )
		{
			const hkClassMember& mem = klass.getDeclaredMember(i);
			if( mem.hasClass() )
			{
				const hkClass& memKlass = mem.getStructClass();
				verifyReflection(memKlass, typeInfoReg, vtableFlagFromClassName, classDoneFromClassName, klassMemberResult, report, memoryManagedPrefixes, numPrefixes, reportNonMemoryManaged );

				if( klassMemberResult == HK_FAILURE )
				{
					result = HK_FAILURE;
				}

				// if type of member is not serializable?
				if( memKlass.getFlags().allAreSet(hkClass::FLAGS_NOT_SERIALIZABLE) )
				{
					// has the member +nosave/serialized(false)?
					if( !mem.getFlags().allAreSet(hkClassMember::SERIALIZE_IGNORED) )
					{
						report << memKlass.getName() << " class is not serializable, but class member " <<
							klassName << "::m_" << mem.getName() << " is not defined with +nosave or +serialized(false).\n";
						result = HK_FAILURE;
					}
					// embedded member?
					if( mem.getType() == hkClassMember::TYPE_STRUCT )
					{
						report << memKlass.getName() << " class is not serializable, but class member " <<
							klassName << "::m_" << mem.getName() << " is embedded.\n";
						result = HK_FAILURE;
					}
				}
			}
			else if( !mem.getFlags().allAreSet(hkClassMember::SERIALIZE_IGNORED)
					&&
					(
						// generic pointer, void*
						( mem.getType() == hkClassMember::TYPE_POINTER
							&& ( mem.getSubType() == hkClassMember::TYPE_STRUCT || mem.getSubType() == hkClassMember::TYPE_VOID ) )
						||
						// array of generic pointers
						( (mem.getType() == hkClassMember::TYPE_ARRAY || mem.getType() == hkClassMember::TYPE_SIMPLEARRAY)
							&& ( mem.getSubType() == hkClassMember::TYPE_POINTER || mem.getSubType() == hkClassMember::TYPE_VOID ) )
					) )
			{
				report << klass.getName() << "::m_" << mem.getName() << " class member is not serializable, but is not defined with +nosave or +serialized(false).\n";
				result = HK_FAILURE;
			}
			if( !mem.getFlags().allAreSet(hkClassMember::SERIALIZE_IGNORED)
				&& ObjectTestController::getFlagFromClassMemberMap().getWithDefault(&mem, 0) == 0 )
			{
				bool referenceCountedClass = mem.hasClass() && hkReferencedObjectClass.isSuperClass(*mem.getClass());
				bool virtualClass = mem.hasClass() && mem.getStructClass().hasVtable();
				bool pointerMember = (mem.getType() == hkClassMember::TYPE_POINTER
					&& mem.getSubType() == hkClassMember::TYPE_STRUCT);
				bool pointerArrayMember = (mem.getType() == hkClassMember::TYPE_SIMPLEARRAY || mem.getType() == hkClassMember::TYPE_ARRAY)
					&& (mem.getSubType() == hkClassMember::TYPE_POINTER);
				bool variantArrayMember = (mem.getType() == hkClassMember::TYPE_SIMPLEARRAY || mem.getType() == hkClassMember::TYPE_ARRAY)
					&& (mem.getSubType() == hkClassMember::TYPE_VARIANT);
				bool objectArrayMember = (mem.getType() == hkClassMember::TYPE_SIMPLEARRAY || mem.getType() == hkClassMember::TYPE_ARRAY)
					&& (mem.getSubType() == hkClassMember::TYPE_STRUCT);
				bool cstringArrayMember = (mem.getType() == hkClassMember::TYPE_SIMPLEARRAY || mem.getType() == hkClassMember::TYPE_ARRAY)
					&& (mem.getSubType() == hkClassMember::TYPE_CSTRING);

				hkBool enforceMemoryManaged = false;
				if( memoryManagedPrefixes )
				{
					for (int prefixIdx = 0; prefixIdx<numPrefixes; prefixIdx++)
					{
						enforceMemoryManaged = enforceMemoryManaged || hkString::beginsWith( klass.getName(), memoryManagedPrefixes[ prefixIdx ] );
					}
				}

				if( mem.getType() == hkClassMember::TYPE_CSTRING
					|| mem.getType() == hkClassMember::TYPE_VARIANT
					|| mem.getType() == hkClassMember::TYPE_SIMPLEARRAY
					|| variantArrayMember
					|| cstringArrayMember
					|| ((pointerMember || pointerArrayMember) && !referenceCountedClass)
					|| (objectArrayMember && virtualClass) )
				{
					ObjectTestController::getFlagFromClassMemberMap().insert(&mem, 1);
					const hkClassEnum* e = hkClassMemberClass.getDeclaredEnumByName("Type");
					const char* typeName;
					const char* subTypeName;
					e->getNameOfValue(mem.getType(), &typeName);
					e->getNameOfValue(mem.getSubType(), &subTypeName);
					if(reportNonMemoryManaged || enforceMemoryManaged)
					{
						report.printf("Non-memory managed\t%s::m_%s\t%s/%s", klassName, mem.getName(), typeName, subTypeName);
						if( mem.hasClass() )
						{
							report.printf("\t%s (%s)", mem.getStructClass().getName(), virtualClass ? "virtual" : "non-virtual");
						}
						report.printf("\n");
					}
					if(enforceMemoryManaged)
					{
						result = HK_FAILURE;
					}
				}
			}
		}
		classDoneFromClassName.insert(klassName, klassMemberResult == HK_SUCCESS ? HK_SUCCESS : HK_FAILURE );
		if( klassMemberResult == HK_SUCCESS && !mayBeAbstract && klassTypeInfo != HK_NULL ) // klass is not abstract and type info is presented, check the finish constructor
		{
			hkLocalArray<char> obj(klass.getObjectSize());
			hkString::memSet(obj.begin(), 0, klass.getObjectSize());

			const hkTypeInfo* typeInfo = typeInfoReg.getTypeInfo(klassName);
			HK_ASSERT(0x41407a66, typeInfo);
			
			// check class members
			ObjectTestController testStructObject;

			// set members
			testStructObject.set(obj.begin(), &klass, HK_NULL, typeInfoReg);

			// finish
			// 
			// we don't need to run finish code
			// just check if data was change by default constructors
			// where we expect finish constructors to be called.
		
			typeInfo->finishLoadedObjectWithoutTracker(obj.begin(), 0);

			// check the finish constructor results
			hkResult finishLoadResult = testStructObject.process(obj.begin(), &klass, HK_NULL, typeInfoReg, report);
			
			if (finishLoadResult == HK_FAILURE)
				result = HK_FAILURE;

		
			// finish
			if( 0 && result == HK_SUCCESS)
			{
				// run clean up function to check reference counting issue
				typeInfo->cleanupLoadedObject(obj.begin());

				result = testStructObject.postprocess(obj.begin(), &klass, HK_NULL, typeInfoReg, report);
			}
		}
	}
}

static void getVtableRegistryMap(const hkClassNameRegistry& classRegistry, const hkTypeInfoRegistry& typeInfoRegistry, hkStringMap<int>& vtableRegistryResultFromClassName, hkResult& result, hkOstream& report)
{
	// collect registered vtables
	const hkVtableClassRegistry* vtableRegistry = hkBuiltinTypeRegistry::getInstance().getVtableClassRegistry();
	HK_ASSERT(0x3bcd08b2, vtableRegistry);
	hkArray<const hkClass*> classes;
	vtableRegistry->getClasses(classes);
	hkResult klassRegistryResult = HK_SUCCESS;
	for( int i = 0; i < classes.getSize(); ++i )
	{
		const char* className = classes[i]->getName();
		klassRegistryResult = HK_SUCCESS;
		if( classRegistry.getClassByName(className) == HK_NULL )
		{
			klassRegistryResult = result = HK_FAILURE;
			report.printf("Class %s has vtable registered, but is missing in the class name registry. Have you registered the class?\n", className);
		}
		if( typeInfoRegistry.getTypeInfo(className) == HK_NULL )
		{
			klassRegistryResult = result = HK_FAILURE;
			report.printf("Class %s has vtable registered, but is missing in the type info registry. Have you registered the class?\n", className);
		}
		vtableRegistryResultFromClassName.insert(className, klassRegistryResult == HK_FAILURE );
	}
}

hkResult HK_CALL hkSerializationCheckingUtils::verifyReflection(const hkClassNameRegistry& classRegistry, hkOstream& report, const char** memoryManagedPrefixes, int numPrefixes, bool reportNonMemoryManaged )
{
	hkResult result = HK_SUCCESS;
	const hkTypeInfoRegistry* typeInfoReg = hkBuiltinTypeRegistry::getInstance().getLoadedObjectRegistry();
	HK_ASSERT(0x3bcd08b1, typeInfoReg);

	// collect registered vtables
	hkStringMap<int> vtableRegistryResultFromClassName;
	getVtableRegistryMap(classRegistry, *typeInfoReg, vtableRegistryResultFromClassName, result, report);

	hkArray<const hkClass*> classes;
	classRegistry.getClasses(classes);
	hkStringMap<int> classDoneFromClassName;

	// Attributes are static, don't need to be checked for mem leaks
	classDoneFromClassName.insert("hkDemoReplayUtilityReplayData", HK_SUCCESS);
	classDoneFromClassName.insert("hkCustomAttributesAttribute", HK_SUCCESS);
	classDoneFromClassName.insert("hkDataObjectTypeAttribute", HK_SUCCESS);
	classDoneFromClassName.insert("hkDocumentationAttribute", HK_SUCCESS);
	classDoneFromClassName.insert("hkDescriptionAttribute", HK_SUCCESS);
	classDoneFromClassName.insert("hkGizmoAttribute", HK_SUCCESS);
	classDoneFromClassName.insert("hkUiAttribute", HK_SUCCESS);

	// Reflection classes are static, ditto.
	classDoneFromClassName.insert("hkCustomAttributes", HK_SUCCESS);
	classDoneFromClassName.insert("hkClassEnumItem", HK_SUCCESS);
	classDoneFromClassName.insert("hkClassMember", HK_SUCCESS);
	classDoneFromClassName.insert("hkClassEnum", HK_SUCCESS);
	classDoneFromClassName.insert("hkClass", HK_SUCCESS);
	
	for( int i = 0; i < classes.getSize(); ++i )
	{
		hkResult classresult = HK_SUCCESS;
		verifyReflection(*classes[i], *typeInfoReg, vtableRegistryResultFromClassName, classDoneFromClassName, classresult, report, memoryManagedPrefixes, numPrefixes, reportNonMemoryManaged );
		if( classresult == HK_FAILURE )
		{
			result = HK_FAILURE;
		}
	}
	ObjectTestController::clearFlagFromClassMemberMap();
	return result;
}

//
// hkSerializationCheckingUtils::DeferredErrorStream
//

int hkSerializationCheckingUtils::DeferredErrorStream::write( const void* buf, int nbytes )
{
	m_data.insertAt( m_data.getSize(), (char*)buf, nbytes ); return nbytes;
}

hkBool hkSerializationCheckingUtils::DeferredErrorStream::isOk() const
{
	return true;
}

void hkSerializationCheckingUtils::DeferredErrorStream::clear()
{
	m_data.clear();
}

void hkSerializationCheckingUtils::DeferredErrorStream::dump()
{
	m_data.pushBack('\0'); // ensure termination

	hkStringBuf messages( m_data.begin() );
	hkArray<const char*>::Temp splits; messages.split('\n', splits );

	for (int i=0; i<splits.getSize(); i++)
	{
		if ( hkString::strLen(splits[i]) )
		{
			hkError::getInstance().message(hkError::MESSAGE_WARNING, 1, splits[i], __FILE__, __LINE__);
		}
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
