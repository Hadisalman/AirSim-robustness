// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/CapsuleComponent.h"
#include "Components/SkeletalMeshComponent.h"
#include "BaseNewerPedestrian.generated.h"



UCLASS()
class ABaseNewerPedestrian : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ABaseNewerPedestrian();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

	UPROPERTY(VisibleDefaultsOnly, Category=Mesh)
	USkeletalMeshComponent* SkeletalMesh;

	//UPROPERTY()
	//UCapsuleComponent* CapsuleComponent;

	UPROPERTY()
	FVector current_goal_position;

	UPROPERTY()
	int32 desired_speed;

	UPROPERTY()
	bool bIsMoving;

	UPROPERTY()
	bool bInCollision;

	UPROPERTY()
	bool bHasCollided;

private:
	TArray<AActor*> OverlapList;


public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	UFUNCTION(BlueprintCallable, Category=Pedestrian)
	void move(FVector goal_position, int32 speed);

	UFUNCTION(BlueprintCallable, Category=Pedestrian)
	void stop();

	UFUNCTION(BlueprintCallable, Category = Pedestrian)
	int32 GetSpeed() const;

	UFUNCTION(BlueprintCallable, Category = Pedestrian)
	bool GetIsMoving() const;
	
	UFUNCTION(BlueprintCallable, Category = Pedestrian)
	bool GetIsInCollision() const;

	UFUNCTION(BlueprintCallable, Category = Pedestrian)
	bool GetHasCollided() const;
	
};
