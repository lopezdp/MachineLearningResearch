# BMI Calculator

# Method used for numeric Input Validation
def checkInt(num):
	# Loop through input to validate data
	while True:
		# Try to convert num to int
		try:
			num = int(num)
			# if num greater than 0
			# then break and reurn num as an int
			if num > 0:
				break
			# else prompt user to enter a num greater than 0
			else:
				num = input("Please enter number greater than zero: ")
		# if try fails catch error
		# and prompt user to enter a number and not a str
		except ValueError:
			print("This is not an int!")
			num = input("Please enter number greater than zero and not a letter: ")
	# output and return the rentalPeriod
	return int(num)

height = checkInt(input("What is your height in cm? "))
weight = checkInt(input("What is your weight in kg? "))

height = height * .01

bmi = weight /  height ** 2
print("\n\tBMI Calculator")
print("\t==============")
print("This is your height: " + str(height))
print("This is your weight: " + str(weight))
print("This is your bmi: " + str(bmi))

